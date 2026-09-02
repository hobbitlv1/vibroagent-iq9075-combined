from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from transformers import get_cosine_schedule_with_warmup

from ..config import resolve_from_config
from ..evaluation.selection import _selection_ids
from ..model.pipeline import VibroGemmaForClassification
from ..utils import set_global_seed
from .checkpoint import TrainerState, load_checkpoint, save_checkpoint
from .data import build_episode_loader, set_loader_epoch, stage_training_settings
from .lm_train import _evaluate_lm


def _load_champion(model: VibroGemmaForClassification, checkpoint: str | Path) -> None:
    source = Path(checkpoint)
    model.load_components(source, strict=False)
    adapter_path = source / "adapter"
    if adapter_path.exists():
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required to load the immutable champion") from exc
        model.gemma = PeftModel.from_pretrained(
            model.gemma,
            adapter_path,
            is_trainable=False,
        )
        model.structured_generator.model = model.gemma
    elif (source / "gemma_trainable.safetensors").exists():
        model.gemma.load_state_dict(
            load_file(str(source / "gemma_trainable.safetensors")),
            strict=False,
        )
    else:
        raise FileNotFoundError(f"Champion checkpoint has no Gemma adapter state: {source}")


def _frozen_eval_train_adapter(model: VibroGemmaForClassification) -> None:
    model.eval()
    if model.common_mode_residual_adapter is None:
        raise RuntimeError("common-mode residual adapter is disabled")
    model.common_mode_residual_adapter.train()


def run_common_mode_residual_training(
    config: dict[str, Any],
    *,
    champion: str | Path,
    resume: str | None = None,
) -> dict[str, Any]:
    """Train only the zero-gated common-mode residual adapter.

    Gemma, the champion LoRA, the original vibration tokenizer and the projector
    remain frozen. The adapter can therefore be discarded without modifying the
    immutable champion.
    """

    set_global_seed(int(config["project"].get("seed", 0)))
    training = config["training"]
    stage = "residual_adapter"
    stage_cfg = dict(training[stage])
    runtime = stage_training_settings(config, stage)
    accumulation = int(runtime["gradient_accumulation_steps"])
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation,
        mixed_precision=str(training.get("mixed_precision", "no")),
    )
    model = VibroGemmaForClassification.from_config(config, training=True)
    initialization = Path(resume) if resume is not None else Path(champion)
    _load_champion(model, initialization)
    if model.common_mode_residual_adapter is None:
        raise RuntimeError("model.common_mode_residual.enabled must be true")

    if resume is None and bool(stage_cfg.get("initialize_from_base_physics_encoder", True)):
        model.common_mode_residual_adapter.encoder.load_state_dict(
            model.vibration_tokenizer.physics_encoder.state_dict(),
            strict=True,
        )
    model.freeze_for_common_mode_adapter()
    if hasattr(model.gemma, "gradient_checkpointing_enable") and bool(
        training.get("gradient_checkpointing", False)
    ):
        model.gemma.gradient_checkpointing_enable()
    if hasattr(model.gemma.config, "use_cache"):
        model.gemma.config.use_cache = False
    model.to(accelerator.device)

    adapter = model.common_mode_residual_adapter
    gate_parameters = [adapter.gate_logits]
    gate_ids = {id(parameter) for parameter in gate_parameters}
    adapter_parameters = [
        parameter
        for parameter in adapter.parameters()
        if parameter.requires_grad and id(parameter) not in gate_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": adapter_parameters,
                "lr": float(stage_cfg.get("learning_rate", 1e-4)),
            },
            {
                "params": gate_parameters,
                "lr": float(stage_cfg.get("gate_learning_rate", 5e-3)),
                "weight_decay": 0.0,
            },
        ],
        weight_decay=float(stage_cfg.get("weight_decay", 1e-4)),
    )

    checkpoint_root = (
        resolve_from_config(config, config["project"]["checkpoint_dir"]) / stage
    )
    validation_episode_ids: set[str] | None = None
    validation_manifest: dict[str, Any] | None = None
    validation_per_stratum = int(stage_cfg.get("validation_episodes_per_stratum", 0))
    if validation_per_stratum > 0:
        index_path = resolve_from_config(config, config["data"]["episode_root"]) / "index.jsonl"
        validation_episode_ids, validation_manifest = _selection_ids(
            index_path,
            split="validation",
            episodes_per_stratum=validation_per_stratum,
            seed=int(config["project"].get("seed", 0)) + 304,
        )
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        (checkpoint_root / "validation_probe_manifest.json").write_text(
            json.dumps(validation_manifest, indent=2),
            encoding="utf-8",
        )

    train_loader = build_episode_loader(
        config,
        "train",
        shuffle=True,
        stage=stage,
        prepare_multiresolution=True,
    )
    validation_loader = build_episode_loader(
        config,
        "validation",
        shuffle=False,
        stage=stage,
        prepare_multiresolution=True,
        episode_ids=validation_episode_ids,
    )
    raw_sampler = getattr(train_loader, "sampler", None)
    sampler_report = raw_sampler.report() if hasattr(raw_sampler, "report") else {
        "strategy": "random_shuffle"
    }

    epochs = int(stage_cfg.get("epochs", 3))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulation))
    total_steps = steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(stage_cfg.get("warmup_ratio", 0.05))),
        num_training_steps=total_steps,
    )
    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        validation_loader,
        scheduler,
    )

    state = TrainerState(stage=stage)
    if resume is not None and (Path(resume) / "trainer_state.json").exists():
        payload = json.loads((Path(resume) / "trainer_state.json").read_text(encoding="utf-8"))
        if str(payload.get("stage")) == stage:
            state = load_checkpoint(
                resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                accelerator=accelerator,
            )

    correction_weight = float(stage_cfg.get("correction_l2_weight", 1e-4))
    gate_weight = float(stage_cfg.get("gate_l1_weight", 1e-3))
    log_every = max(1, int(stage_cfg.get("log_every_steps", training.get("log_every_steps", 25))))

    accelerator.print(
        json.dumps(
            {
                "stage": stage,
                "champion": str(Path(champion).resolve()),
                "resume": resume,
                "epochs": epochs,
                "start_epoch": state.epoch + 1,
                "batch_size": runtime["batch_size"],
                "gradient_accumulation_steps": accumulation,
                "effective_batch_size_per_process": runtime["batch_size"] * accumulation,
                "optimizer_steps_per_epoch": steps_per_epoch,
                "sampling": sampler_report,
                "validation_probe": validation_manifest,
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in accelerator.unwrap_model(model).parameters()
                    if parameter.requires_grad
                ),
                "correction_l2_weight": correction_weight,
                "gate_l1_weight": gate_weight,
            },
            sort_keys=True,
        )
    )

    for epoch in range(state.epoch, epochs):
        set_loader_epoch(train_loader, epoch)
        unwrapped = accelerator.unwrap_model(model)
        _frozen_eval_train_adapter(unwrapped)
        started = time.perf_counter()
        interval_started = started
        interval_examples = 0
        last_values: dict[str, float] = {}
        optimizer.zero_grad(set_to_none=True)

        for prepared_batch in train_loader:
            interval_examples += len(prepared_batch)
            with accelerator.accumulate(model):
                output = model(prepared_batch)
                language_loss = output.loss
                if language_loss is None or not torch.isfinite(language_loss):
                    raise RuntimeError(
                        f"non-finite residual-adapter language loss at step {state.global_step}: {language_loss}"
                    )
                correction_energy, gate_magnitude = unwrapped.common_mode_regularization()
                loss = (
                    language_loss
                    + correction_weight * correction_energy
                    + gate_weight * gate_magnitude
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unwrapped.common_mode_residual_adapter.parameters(),
                        float(training.get("max_grad_norm", 1.0)),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                state.global_step += 1
                last_values = {
                    "loss": float(loss.detach().float().item()),
                    "language_loss": float(language_loss.detach().float().item()),
                    "correction_energy": float(correction_energy.detach().float().item()),
                    "gate_magnitude": float(gate_magnitude.detach().float().item()),
                }
                if state.global_step % log_every == 0:
                    now = time.perf_counter()
                    elapsed = max(now - interval_started, 1e-9)
                    accelerator.print(
                        json.dumps(
                            {
                                "stage": stage,
                                "event": "progress",
                                "epoch": epoch + 1,
                                "epochs": epochs,
                                "global_step": state.global_step,
                                "optimizer_steps_per_epoch": steps_per_epoch,
                                **last_values,
                                "gate_values": [
                                    float(value)
                                    for value in unwrapped.common_mode_residual_adapter.gate_values.detach().cpu().tolist()
                                ],
                                "learning_rate": [float(value) for value in scheduler.get_last_lr()],
                                "examples_per_second": interval_examples / elapsed,
                                "elapsed_epoch_minutes": (now - started) / 60.0,
                            },
                            sort_keys=True,
                        )
                    )
                    interval_started = now
                    interval_examples = 0

        unwrapped.eval()
        validation_started = time.perf_counter()
        validation_loss = _evaluate_lm(accelerator, model, validation_loader)
        validation_seconds = time.perf_counter() - validation_started
        epoch_seconds = time.perf_counter() - started
        state.epoch = epoch + 1
        improved = state.best_validation_loss is None or validation_loss < state.best_validation_loss
        if improved:
            state.best_validation_loss = validation_loss
            state.epochs_without_improvement = 0
        else:
            state.epochs_without_improvement += 1
        gate_values = [
            float(value)
            for value in unwrapped.common_mode_residual_adapter.gate_values.detach().cpu().tolist()
        ]
        residual_summary = {
            "gate_values": gate_values,
            "maximum_absolute_gate": max((abs(value) for value in gate_values), default=0.0),
            "correction_rms": float(
                max(float(last_values.get("correction_energy", 0.0)), 0.0) ** 0.5
            ),
        }
        extra = {
            "champion": str(Path(champion).resolve()),
            "validation_loss": validation_loss,
            "epoch_seconds": epoch_seconds,
            "validation_seconds": validation_seconds,
            "last_training_values": last_values,
            "gate_values": gate_values,
            "residual": residual_summary,
            "sampling": sampler_report,
            "validation_probe": validation_manifest,
        }
        if accelerator.is_main_process:
            save_checkpoint(
                checkpoint_root / f"epoch-{epoch + 1:03d}",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
                accelerator=accelerator,
                extra=extra,
            )
            if improved:
                save_checkpoint(
                    checkpoint_root / "best",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=state,
                    accelerator=accelerator,
                    extra=extra,
                )
        accelerator.print(
            json.dumps(
                {
                    "stage": stage,
                    "event": "epoch_end",
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "global_step": state.global_step,
                    "validation_loss": validation_loss,
                    "best_validation_loss": state.best_validation_loss,
                    "improved": improved,
                    "gate_values": extra["gate_values"],
                    "epoch_minutes": epoch_seconds / 60.0,
                    "validation_minutes": validation_seconds / 60.0,
                },
                sort_keys=True,
            )
        )

    state.completed = True
    state.stop_reason = "maximum_epochs_reached"
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        save_checkpoint(
            checkpoint_root / "final",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            accelerator=accelerator,
            extra={
                "champion": str(Path(champion).resolve()),
                "gate_values": [
                    float(value)
                    for value in unwrapped.common_mode_residual_adapter.gate_values.detach().cpu().tolist()
                ],
                "residual": {
                    "gate_values": [
                        float(value)
                        for value in unwrapped.common_mode_residual_adapter.gate_values.detach().cpu().tolist()
                    ],
                    "maximum_absolute_gate": float(
                        unwrapped.common_mode_residual_adapter.gate_values.detach().float().abs().max().item()
                    ),
                },
                "sampling": sampler_report,
                "validation_probe": validation_manifest,
            },
        )
    return {
        "stage": stage,
        "champion": str(Path(champion).resolve()),
        "checkpoint_root": str(checkpoint_root),
        "epochs_completed": state.epoch,
        "global_step": state.global_step,
        "best_validation_loss": state.best_validation_loss,
        "gate_values": [
            float(value)
            for value in unwrapped.common_mode_residual_adapter.gate_values.detach().cpu().tolist()
        ],
    }
