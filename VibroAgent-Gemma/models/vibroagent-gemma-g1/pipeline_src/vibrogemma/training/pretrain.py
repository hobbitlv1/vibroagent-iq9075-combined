from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

import torch
from accelerate import Accelerator
from torch import nn
from torch.nn import functional as F
from transformers import get_cosine_schedule_with_warmup

from ..config import resolve_from_config, validate_training_controls
from ..contracts import SixBoardEpisode
from ..model.encoder import MultiBranchReconstructionHead, SixBoardVibrationTokenizer
from ..signal.multiresolution import (
    MultiResolutionEpisodePreprocessor,
    PreparedEpisode,
    PreparedEpisodeBatch,
    prepared_episode_batch,
)
from ..utils import set_global_seed
from .checkpoint import TrainerState, load_checkpoint, save_checkpoint
from .data import build_episode_loader, set_loader_epoch, stage_training_settings


def _loader_corpus_report(loader: Any) -> dict[str, Any]:
    records = list(getattr(getattr(loader, "dataset", None), "records", []))
    return {
        "windows": len(records),
        "datasets": dict(
            sorted(Counter(str(row.get("dataset_id", "")) for row in records).items())
        ),
        "training_roles": dict(
            sorted(
                Counter(str(row.get("training_role", "")) for row in records).items()
            )
        ),
        "labelled_windows": sum(bool(row.get("has_label", True)) for row in records),
        "unlabelled_windows": sum(
            not bool(row.get("has_label", True)) for row in records
        ),
    }


class MultiResolutionPretrainer(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config_payload = config
        model = config["model"]
        multi = model.get("multiresolution", {})
        modal = multi.get("modal", {})
        wideband = multi.get("wideband", {})
        physics = multi.get("physics", {})
        fusion = multi.get("fusion", {})
        self.preprocessor = MultiResolutionEpisodePreprocessor.from_config(config)
        width = int(model.get("encoder_width", 192))
        self.tokenizer = SixBoardVibrationTokenizer(
            width=width,
            modal_depth=int(modal.get("depth", model.get("encoder_depth", 6))),
            modal_tokens_per_axis=int(modal.get("tokens_per_axis", 4)),
            modal_tokens_per_board=int(modal.get("tokens_per_board", 6)),
            modal_kernel_sizes=modal.get(
                "kernel_sizes", model.get("encoder_kernel_sizes", [7, 5, 5, 3, 3, 3])
            ),
            modal_dilations=modal.get(
                "dilations", model.get("encoder_dilations", [1, 2, 4, 8, 16, 32])
            ),
            wideband_input_channels=int(wideband.get("input_channels", 9)),
            wideband_depth=int(wideband.get("depth", 4)),
            wideband_tokens_per_board=int(wideband.get("tokens_per_board", 6)),
            wideband_frequency_patches=int(wideband.get("frequency_patches", 4)),
            wideband_time_patches=int(wideband.get("time_patches", 4)),
            physics_feature_count=len(self.preprocessor.physics_extractor.feature_names),
            physics_tokens_per_board=int(physics.get("tokens_per_board", 2)),
            attention_heads=int(fusion.get("attention_heads", 4)),
            branch_fusion_layers=int(fusion.get("branch_layers", 2)),
            cross_board_layers=int(fusion.get("cross_board_layers", 2)),
            dropout=float(model.get("dropout", 0.05)),
        )
        pretrain = config["training"]["pretrain"]
        self.modal_output_samples = int(pretrain.get("modal_reconstruction_samples", 256))
        self.spectral_output_frequency = int(pretrain.get("spectral_reconstruction_frequency_bins", 32))
        self.spectral_output_time = int(pretrain.get("spectral_reconstruction_time_bins", 16))
        self.reconstruction = MultiBranchReconstructionHead(
            width=width,
            modal_tokens=self.tokenizer.modal_tokens_per_board,
            wideband_tokens=self.tokenizer.wideband_tokens_per_board,
            physics_tokens=self.tokenizer.physics_tokens_per_board,
            modal_output_samples=self.modal_output_samples,
            spectral_output_frequency=self.spectral_output_frequency,
            spectral_output_time=self.spectral_output_time,
            physics_feature_count=len(self.preprocessor.physics_extractor.feature_names),
        )
        self.loss_weights = {
            "modal": float(pretrain.get("modal_loss_weight", 1.0)),
            "wideband": float(pretrain.get("wideband_loss_weight", 1.0)),
            "physics": float(pretrain.get("physics_loss_weight", 0.5)),
        }
        self.reconstruction_token_source = str(
            pretrain.get("reconstruction_token_source", "branch")
        )
        if self.reconstruction_token_source not in {"branch", "fused"}:
            raise ValueError(
                "reconstruction_token_source must be 'branch' or 'fused'"
            )
        self.final_token_consistency_weight = float(
            pretrain.get("final_token_consistency_weight", 0.0)
        )
        if (
            not math.isfinite(self.final_token_consistency_weight)
            or self.final_token_consistency_weight < 0.0
        ):
            raise ValueError(
                "final_token_consistency_weight must be finite and non-negative"
            )
        if (
            self.final_token_consistency_weight > 0.0
            and sum(self.loss_weights.values()) <= 0.0
        ):
            raise ValueError(
                "final-token consistency requires a positive branch reconstruction anchor"
            )
        self.last_loss_components: dict[str, torch.Tensor] | None = None

    @property
    def component_device(self) -> torch.device:
        return next(self.tokenizer.parameters()).device

    def forward(
        self,
        episodes: Sequence[SixBoardEpisode | PreparedEpisode] | PreparedEpisodeBatch,
        mask_ratios: dict[str, float] | float,
    ) -> torch.Tensor:
        if isinstance(mask_ratios, (int, float)):
            ratios = {"modal": float(mask_ratios), "wideband": float(mask_ratios), "physics": float(mask_ratios)}
        else:
            ratios = {
                "modal": float(mask_ratios["modal"]),
                "wideband": float(mask_ratios["wideband"]),
                "physics": float(mask_ratios["physics"]),
            }
        if any(not 0.0 <= value < 1.0 for value in ratios.values()):
            raise ValueError("all branch mask ratios must be in [0,1)")
        prepared = prepared_episode_batch(
            episodes,
            preprocessor=self.preprocessor,
            device=self.component_device,
        )
        batch = prepared.tensors
        modal = batch["modal_values"]
        modal_valid = batch["modal_sample_mask"].unsqueeze(2) & batch["modal_axis_mask"].unsqueeze(-1)
        modal_corruption = (
            torch.rand((modal.shape[0], modal.shape[1], 1, modal.shape[-1]), device=modal.device) < ratios["modal"]
        ).expand_as(modal) & modal_valid
        corrupted_modal = modal.masked_fill(modal_corruption, 0.0)

        wideband = batch["wideband_features"]
        wideband_corruption = (
            torch.rand(
                (wideband.shape[0], wideband.shape[1], 1, 1, wideband.shape[-2], wideband.shape[-1]),
                device=wideband.device,
            )
            < ratios["wideband"]
        )
        corrupted_wideband = wideband.clone()
        # Channels 0..6 contain measurements; channels 7..8 are validity flags.
        corrupted_wideband[:, :, :, :7] = corrupted_wideband[:, :, :, :7].masked_fill(
            wideband_corruption.expand(-1, -1, 3, 7, -1, -1), 0.0
        )

        physics = batch["physics_features"]
        physics_available = batch["physics_mask"]
        physics_corruption = (torch.rand_like(physics) < ratios["physics"]) & physics_available
        corrupted_physics = physics.masked_fill(physics_corruption, 0.0)
        corrupted_physics_mask = physics_available & ~physics_corruption

        tokenizer_output = self.tokenizer(
            modal_values=corrupted_modal,
            modal_axis_mask=batch["modal_axis_mask"],
            modal_sample_mask=batch["modal_sample_mask"],
            wideband_features=corrupted_wideband,
            wideband_frequency_mask=batch["wideband_frequency_mask"],
            wideband_cross_valid=batch["wideband_cross_valid"],
            physics_features=corrupted_physics,
            physics_mask=corrupted_physics_mask,
            sensor_positions=batch["sensor_positions"],
            orientation_features=batch["orientation_features"],
            quality_features=batch["quality_features"],
        )
        final_token_consistency = tokenizer_output.tokens.new_zeros(())
        if self.final_token_consistency_weight > 0.0:
            tokenizer_was_training = self.tokenizer.training
            self.tokenizer.eval()
            try:
                with torch.no_grad():
                    clean_teacher = self.tokenizer(
                        modal_values=modal,
                        modal_axis_mask=batch["modal_axis_mask"],
                        modal_sample_mask=batch["modal_sample_mask"],
                        wideband_features=wideband,
                        wideband_frequency_mask=batch["wideband_frequency_mask"],
                        wideband_cross_valid=batch["wideband_cross_valid"],
                        physics_features=physics,
                        physics_mask=physics_available,
                        sensor_positions=batch["sensor_positions"],
                        orientation_features=batch["orientation_features"],
                        quality_features=batch["quality_features"],
                    )
            finally:
                self.tokenizer.train(tokenizer_was_training)
            consistency_mask = tokenizer_output.token_mask & clean_teacher.token_mask
            if not consistency_mask.any():
                raise RuntimeError("final-token consistency found no mutually valid tokens")
            token_error = F.smooth_l1_loss(
                tokenizer_output.tokens.float(),
                clean_teacher.tokens.detach().float(),
                reduction="none",
            ).mean(dim=-1)
            final_token_consistency = token_error[consistency_mask].mean()
        reconstructed_modal, reconstructed_wideband, reconstructed_physics = (
            self.reconstruction(
                tokenizer_output,
                token_source=self.reconstruction_token_source,
            )
        )

        bsz, boards, _, modal_samples = modal.shape
        modal_target = F.adaptive_avg_pool1d(
            modal.reshape(bsz * boards, 3, modal_samples), self.modal_output_samples
        ).reshape(bsz, boards, 3, self.modal_output_samples)
        modal_corruption_pooled = F.adaptive_max_pool1d(
            modal_corruption.float().reshape(bsz * boards * 3, 1, modal_samples),
            self.modal_output_samples,
        ).reshape(bsz, boards, 3, self.modal_output_samples) > 0
        modal_valid_pooled = F.adaptive_max_pool1d(
            modal_valid.float().reshape(bsz * boards * 3, 1, modal_samples),
            self.modal_output_samples,
        ).reshape(bsz, boards, 3, self.modal_output_samples) > 0
        modal_loss_mask = modal_corruption_pooled & modal_valid_pooled
        if not modal_loss_mask.any():
            modal_loss_mask = modal_valid_pooled
        modal_loss = F.smooth_l1_loss(
            reconstructed_modal[modal_loss_mask], modal_target[modal_loss_mask]
        )

        log_power = wideband[:, :, :, 0]
        wide_target = F.adaptive_avg_pool2d(
            log_power.reshape(bsz * boards, 3, log_power.shape[-2], log_power.shape[-1]),
            (self.spectral_output_frequency, self.spectral_output_time),
        ).reshape(bsz, boards, 3, self.spectral_output_frequency, self.spectral_output_time)
        wide_corruption_axis = wideband_corruption.expand(-1, -1, 3, 1, -1, -1)[:, :, :, 0]
        wide_corruption_pooled = F.adaptive_max_pool2d(
            wide_corruption_axis.float().reshape(
                bsz * boards, 3, wide_corruption_axis.shape[-2], wide_corruption_axis.shape[-1]
            ),
            (self.spectral_output_frequency, self.spectral_output_time),
        ).reshape(bsz, boards, 3, self.spectral_output_frequency, self.spectral_output_time) > 0
        wide_valid = (
            batch["modal_axis_mask"][:, :, :, None, None]
            & batch["wideband_frequency_mask"][:, :, None, :, None]
        )
        wide_valid = wide_valid.expand(-1, -1, -1, -1, log_power.shape[-1])
        wide_valid_pooled = F.adaptive_max_pool2d(
            wide_valid.float().reshape(bsz * boards, 3, wide_valid.shape[-2], wide_valid.shape[-1]),
            (self.spectral_output_frequency, self.spectral_output_time),
        ).reshape(bsz, boards, 3, self.spectral_output_frequency, self.spectral_output_time) > 0
        wide_loss_mask = wide_corruption_pooled & wide_valid_pooled
        if not wide_loss_mask.any():
            wide_loss_mask = wide_valid_pooled
        wideband_loss = F.smooth_l1_loss(
            reconstructed_wideband[wide_loss_mask], wide_target[wide_loss_mask]
        )

        physics_loss_mask = physics_corruption
        if not physics_loss_mask.any():
            physics_loss_mask = physics_available
        physics_loss = F.smooth_l1_loss(
            reconstructed_physics[physics_loss_mask], physics[physics_loss_mask]
        )
        branch_reconstruction = (
            self.loss_weights["modal"] * modal_loss
            + self.loss_weights["wideband"] * wideband_loss
            + self.loss_weights["physics"] * physics_loss
        )
        total = branch_reconstruction
        if self.final_token_consistency_weight > 0.0:
            total = total + (
                self.final_token_consistency_weight * final_token_consistency
            )
        self.last_loss_components = {
            "modal_reconstruction": modal_loss,
            "wideband_reconstruction": wideband_loss,
            "physics_reconstruction": physics_loss,
            "branch_reconstruction_weighted": branch_reconstruction,
            "final_token_consistency": final_token_consistency,
            "total": total,
        }
        return total


def run_pretraining(config: dict[str, Any], *, resume: str | None = None) -> dict[str, Any]:
    validate_training_controls(config)
    set_global_seed(int(config["project"].get("seed", 0)))
    training = config["training"]
    stage_cfg = training["pretrain"]
    runtime = stage_training_settings(config, "pretrain")
    gradient_accumulation_steps = int(runtime["gradient_accumulation_steps"])
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=str(training.get("mixed_precision", "no")),
    )
    model = MultiResolutionPretrainer(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stage_cfg["learning_rate"]),
        weight_decay=float(stage_cfg.get("weight_decay", 0.0)),
    )
    train_loader = build_episode_loader(
        config,
        "train",
        shuffle=True,
        require_labels=False,
        stage="pretrain",
        prepare_multiresolution=True,
    )
    validation_loader = build_episode_loader(
        config,
        "validation",
        shuffle=False,
        require_labels=False,
        stage="pretrain",
        prepare_multiresolution=True,
    )
    raw_sampler = getattr(train_loader, "sampler", None)
    sampler_report = (
        raw_sampler.report()
        if hasattr(raw_sampler, "report")
        else {"strategy": "random_shuffle"}
    )
    epoch_aware_sampling = hasattr(raw_sampler, "set_epoch")
    corpus_report = {
        "configured_filter": dict(stage_cfg.get("corpus") or {}),
        "train": _loader_corpus_report(train_loader),
        "validation": _loader_corpus_report(validation_loader),
    }
    epochs = int(stage_cfg["epochs"])
    optimizer_steps_per_epoch = max(
        1,
        math.ceil(len(train_loader) / gradient_accumulation_steps),
    )
    total_steps = optimizer_steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(stage_cfg.get("warmup_ratio", 0.0))),
        num_training_steps=total_steps,
    )
    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, scheduler
    )
    state = TrainerState(stage="pretrain")
    if resume:
        state = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
        )
    checkpoint_root = resolve_from_config(config, config["project"]["checkpoint_dir"]) / "pretrain"
    log_every = max(1, int(stage_cfg.get("log_every_steps", training.get("log_every_steps", 10))))
    accelerator.print(
        json.dumps(
            {
                "stage": "pretrain",
                "resume": resume,
                "start_epoch": state.epoch + 1,
                "epochs": epochs,
                "batch_size": runtime["batch_size"],
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size_per_process": runtime["batch_size"]
                * gradient_accumulation_steps,
                "num_workers": runtime["num_workers"],
                "validation_num_workers": runtime["validation_num_workers"],
                "prefetch_factor": runtime["prefetch_factor"],
                "worker_thread_limit": runtime["worker_thread_limit"],
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "parallel_preprocessing": True,
                "sampling": sampler_report,
                "corpus": corpus_report,
                "reconstruction_token_source": str(
                    stage_cfg.get("reconstruction_token_source", "branch")
                ),
                "final_token_consistency_weight": float(
                    stage_cfg.get("final_token_consistency_weight", 0.0)
                ),
            },
            sort_keys=True,
        )
    )

    for epoch in range(state.epoch, epochs):
        if epoch_aware_sampling:
            set_loader_epoch(train_loader, epoch)
        model.train()
        epoch_started = time.perf_counter()
        interval_started = epoch_started
        interval_examples = 0
        interval_batches = 0
        last_loss = float("nan")
        last_loss_components: dict[str, float] = {}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        accelerator.print(
            json.dumps(
                {
                    "stage": "pretrain",
                    "event": "epoch_start",
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "global_step": state.global_step,
                },
                sort_keys=True,
            )
        )
        optimizer.zero_grad(set_to_none=True)
        for prepared_batch in train_loader:
            batch_examples = len(prepared_batch)
            interval_examples += batch_examples
            interval_batches += 1
            with accelerator.accumulate(model):
                loss = model(prepared_batch, _mask_ratios(stage_cfg))
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite pretraining loss at step {state.global_step}: {loss}"
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        float(training.get("max_grad_norm", 1.0)),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                state.global_step += 1
                last_loss = float(loss.detach().float().item())
                unwrapped_model = accelerator.unwrap_model(model)
                raw_components = getattr(
                    unwrapped_model, "last_loss_components", None
                )
                if raw_components:
                    last_loss_components = {
                        str(name): float(value.detach().float().item())
                        for name, value in raw_components.items()
                    }
                if state.global_step % log_every == 0:
                    now = time.perf_counter()
                    elapsed = max(now - interval_started, 1e-9)
                    payload = {
                        "stage": "pretrain",
                        "event": "progress",
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "global_step": state.global_step,
                        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                        "loss": last_loss,
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                        "examples_per_second": interval_examples / elapsed,
                        "batches_per_second": interval_batches / elapsed,
                        "elapsed_epoch_minutes": (now - epoch_started) / 60.0,
                    }
                    if last_loss_components:
                        payload["loss_components"] = last_loss_components
                    if torch.cuda.is_available():
                        payload["gpu_peak_memory_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
                    accelerator.print(json.dumps(payload, sort_keys=True))
                    interval_started = now
                    interval_examples = 0
                    interval_batches = 0

        validation_started = time.perf_counter()
        validation_loss = _evaluate_pretrainer(
            accelerator,
            model,
            validation_loader,
            _mask_ratios(stage_cfg),
        )
        validation_seconds = time.perf_counter() - validation_started
        epoch_seconds = time.perf_counter() - epoch_started
        state.epoch = epoch + 1
        checkpoint_extra = {
            "validation_loss": validation_loss,
            "training_loss_last": last_loss,
            "epoch_seconds": epoch_seconds,
            "validation_seconds": validation_seconds,
            "runtime": runtime,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "parallel_preprocessing": True,
            "sampling": sampler_report,
            "corpus": corpus_report,
            "reconstruction_token_source": str(
                stage_cfg.get("reconstruction_token_source", "branch")
            ),
            "loss_components_last": last_loss_components,
        }
        if state.best_validation_loss is None or validation_loss < state.best_validation_loss:
            state.best_validation_loss = validation_loss
            if accelerator.is_main_process:
                save_checkpoint(
                    checkpoint_root / "best",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=state,
                    accelerator=accelerator,
                    extra=checkpoint_extra,
                )
        if accelerator.is_main_process:
            save_checkpoint(
                checkpoint_root / f"epoch-{epoch + 1:03d}",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
                accelerator=accelerator,
                extra=checkpoint_extra,
            )
        accelerator.print(
            json.dumps(
                {
                    "stage": "pretrain",
                    "event": "epoch_end",
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "global_step": state.global_step,
                    "validation_loss": validation_loss,
                    "best_validation_loss": state.best_validation_loss,
                    "epoch_minutes": epoch_seconds / 60.0,
                    "validation_minutes": validation_seconds / 60.0,
                },
                sort_keys=True,
            )
        )
    state.completed = True
    if accelerator.is_main_process:
        save_checkpoint(
            checkpoint_root / "final",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            accelerator=accelerator,
            extra={
                "runtime": runtime,
                "parallel_preprocessing": True,
                "sampling": sampler_report,
                "corpus": corpus_report,
                "reconstruction_token_source": str(
                    stage_cfg.get("reconstruction_token_source", "branch")
                ),
                "final_token_consistency_weight": float(
                    stage_cfg.get("final_token_consistency_weight", 0.0)
                ),
            },
        )
    return {
        "stage": state.stage,
        "global_step": state.global_step,
        "best_validation_loss": state.best_validation_loss,
    }


@torch.inference_mode()
def _mask_ratios(stage_cfg: dict[str, Any]) -> dict[str, float]:
    fallback = float(stage_cfg.get("mask_ratio", 0.35))
    return {
        "modal": float(stage_cfg.get("modal_mask_ratio", fallback)),
        "wideband": float(stage_cfg.get("wideband_mask_ratio", fallback)),
        "physics": float(stage_cfg.get("physics_mask_ratio", fallback)),
    }


@torch.inference_mode()
def _evaluate_pretrainer(
    accelerator: Accelerator,
    model: nn.Module,
    loader,
    mask_ratios: dict[str, float],
) -> float:
    model.eval()
    losses: list[torch.Tensor] = []
    for prepared_batch in loader:
        loss = model(prepared_batch, mask_ratios)
        losses.append(accelerator.gather_for_metrics(loss.detach().reshape(1)))
    if not losses:
        raise RuntimeError("validation loader produced no batches")
    return float(torch.cat(losses).mean().item())
