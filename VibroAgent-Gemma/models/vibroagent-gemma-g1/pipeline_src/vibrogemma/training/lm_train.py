from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Literal

import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from transformers import get_cosine_schedule_with_warmup

from ..config import resolve_from_config, validate_training_controls
from ..model.pipeline import VibroGemmaForClassification
from ..utils import set_global_seed
from .checkpoint import (
    TrainerState,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_data_contract,
)
from .data import build_episode_loader, set_loader_epoch, stage_training_settings

_DECISION_STAT_KEYS = (
    "example_count",
    "global_predicted_anomaly",
    "global_normal_support",
    "global_normal_correct",
    "global_anomaly_support",
    "global_anomaly_correct",
    "single_target_support",
    "single_target_all_zero",
    "single_target_all_five",
    "single_target_predicted_cardinality",
)

def _gemma_optimizer_parameter_groups(
    model: VibroGemmaForClassification,
    stage_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build optimizer groups for the Gemma-only decision architecture.

    Alignment exposes the vibration tokenizer and projector. LoRA adds only the
    already-audited Gemma adapter parameters. Any other trainable module is an
    architectural error rather than an optimizer group to discover implicitly.
    """

    encoder_lr = float(
        stage_cfg.get("encoder_learning_rate", stage_cfg["learning_rate"])
    )
    gemma_lr = float(stage_cfg["learning_rate"])
    encoder_parameters = [
        *[
            parameter
            for parameter in model.vibration_tokenizer.parameters()
            if parameter.requires_grad
        ],
        *[
            parameter
            for parameter in model.projector.parameters()
            if parameter.requires_grad
        ],
    ]
    gemma_parameters = [
        parameter for parameter in model.gemma.parameters() if parameter.requires_grad
    ]
    allowed = {
        id(parameter) for parameter in (*encoder_parameters, *gemma_parameters)
    }
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in allowed
    ]
    if unexpected:
        raise RuntimeError(
            "Gemma-only training found trainable parameters outside "
            f"vibration_tokenizer/projector/Gemma: {unexpected[:10]}"
        )
    groups: list[dict[str, Any]] = []
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": encoder_lr})
    if gemma_parameters:
        groups.append({"params": gemma_parameters, "lr": gemma_lr})
    return groups


def _empty_decision_counts() -> dict[str, float]:
    return {key: 0.0 for key in _DECISION_STAT_KEYS}


def _accumulate_decision_stats(
    accelerator: Accelerator,
    counts: dict[str, float],
    stats: dict[str, torch.Tensor] | None,
) -> None:
    if not stats:
        return
    for key in _DECISION_STAT_KEYS:
        value = stats.get(key)
        if value is None:
            continue
        reduced = accelerator.reduce(value.detach().float(), reduction="sum")
        counts[key] += float(reduced.item())


def _decision_guard_metrics(counts: dict[str, float]) -> dict[str, float | None]:
    examples = counts["example_count"]
    normal_support = counts["global_normal_support"]
    anomaly_support = counts["global_anomaly_support"]
    single_support = counts["single_target_support"]
    return {
        "examples": examples,
        "global_anomaly_prediction_rate": (
            counts["global_predicted_anomaly"] / examples if examples > 0 else None
        ),
        "normal_recall": (
            counts["global_normal_correct"] / normal_support
            if normal_support > 0
            else None
        ),
        "anomaly_recall": (
            counts["global_anomaly_correct"] / anomaly_support
            if anomaly_support > 0
            else None
        ),
        "single_target_all_zero_rate": (
            counts["single_target_all_zero"] / single_support
            if single_support > 0
            else None
        ),
        "single_target_all_five_rate": (
            counts["single_target_all_five"] / single_support
            if single_support > 0
            else None
        ),
        "mean_single_target_predicted_cardinality": (
            counts["single_target_predicted_cardinality"] / single_support
            if single_support > 0
            else None
        ),
    }


def _guard_failures(
    metrics: dict[str, float | None], guard: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    anomaly_rate = metrics["global_anomaly_prediction_rate"]
    if anomaly_rate is not None:
        maximum = float(guard.get("maximum_global_anomaly_prediction_rate", 0.90))
        minimum = float(guard.get("minimum_global_anomaly_prediction_rate", 0.10))
        if anomaly_rate > maximum:
            failures.append(
                f"global anomaly prediction rate {anomaly_rate:.4f} > {maximum:.4f}"
            )
        if anomaly_rate < minimum:
            failures.append(
                f"global anomaly prediction rate {anomaly_rate:.4f} < {minimum:.4f}"
            )
    for name, config_name, default in (
        ("normal_recall", "minimum_normal_recall", 0.50),
        ("anomaly_recall", "minimum_anomaly_recall", 0.50),
    ):
        value = metrics[name]
        threshold = float(guard.get(config_name, default))
        if value is not None and value < threshold:
            failures.append(f"{name} {value:.4f} < {threshold:.4f}")
    for name, config_name, default in (
        ("single_target_all_zero_rate", "maximum_single_target_all_zero_rate", 0.10),
        ("single_target_all_five_rate", "maximum_single_target_all_five_rate", 0.10),
        (
            "mean_single_target_predicted_cardinality",
            "maximum_mean_single_target_predicted_cardinality",
            1.50,
        ),
    ):
        value = metrics[name]
        threshold = float(guard.get(config_name, default))
        if value is not None and value > threshold:
            failures.append(f"{name} {value:.4f} > {threshold:.4f}")
    return failures


def _load_pretrain_encoder(
    model: VibroGemmaForClassification,
    source: str | Path,
    *,
    initialize_projector: bool = True,
) -> None:
    """Load encoder lineage, optionally retaining a newly initialized projector."""

    root = Path(source)
    if (root / "model.safetensors").exists():
        state = load_file(str(root / "model.safetensors"))
        prefix = "tokenizer."
        encoder_state = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
        if not encoder_state:
            raise RuntimeError(f"pretraining checkpoint has no tokenizer weights: {root}")
        model.vibration_tokenizer.load_state_dict(encoder_state, strict=False)
    elif (root / "vibration_components.safetensors").exists():
        if initialize_projector:
            model.load_components(root, strict=False)
        else:
            state = load_file(str(root / "vibration_components.safetensors"))
            prefix = "vibration_tokenizer."
            encoder_state = {
                key.removeprefix(prefix): value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            if not encoder_state:
                raise RuntimeError(
                    f"vibration checkpoint has no encoder weights: {root}"
                )
            model.vibration_tokenizer.load_state_dict(encoder_state, strict=False)
    else:
        raise FileNotFoundError(f"No compatible pretraining/alignment state in {root}")



def _checkpoint_stage(source: str | Path | None) -> str | None:
    if source is None:
        return None
    state_path = Path(source) / "trainer_state.json"
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return str(payload.get("stage")) if payload.get("stage") is not None else None


def _load_lora_initialization(
    model: VibroGemmaForClassification, source: str | Path
) -> None:
    """Load modality components and LoRA weights without optimizer state."""

    root = Path(source)
    _load_pretrain_encoder(model, root)
    trainable_path = root / "gemma_trainable.safetensors"
    if trainable_path.exists():
        model.gemma.load_state_dict(load_file(str(trainable_path)), strict=False)
        return
    adapter_path = root / "adapter"
    if adapter_path.exists() and hasattr(model.gemma, "load_adapter"):
        try:
            model.gemma.load_adapter(
                str(adapter_path), adapter_name="default", is_trainable=True
            )
            return
        except (TypeError, ValueError):
            pass
    raise FileNotFoundError(f"No LoRA adapter state in {root}")


def _configure_downstream_trainability(
    model: VibroGemmaForClassification,
    *,
    stage: Literal["align", "lora"],
    stage_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Apply explicit encoder freezing while preserving legacy LoRA controls."""

    freeze_encoder = bool(stage_cfg.get("freeze_encoder", False))
    freeze_all_vibration = bool(
        stage == "lora" and not stage_cfg.get("train_vibration_components", True)
    )
    if freeze_encoder or freeze_all_vibration:
        for parameter in model.vibration_tokenizer.parameters():
            parameter.requires_grad = False
    if freeze_all_vibration:
        for parameter in model.projector.parameters():
            parameter.requires_grad = False
    return {
        "freeze_encoder": freeze_encoder or freeze_all_vibration,
        "freeze_projector": freeze_all_vibration,
        "encoder_trainable_parameters": sum(
            parameter.numel()
            for parameter in model.vibration_tokenizer.parameters()
            if parameter.requires_grad
        ),
        "projector_trainable_parameters": sum(
            parameter.numel()
            for parameter in model.projector.parameters()
            if parameter.requires_grad
        ),
    }


def _enforce_frozen_encoder_eval(model: VibroGemmaForClassification) -> None:
    if not any(
        parameter.requires_grad for parameter in model.vibration_tokenizer.parameters()
    ):
        model.vibration_tokenizer.eval()

def _apply_lora(model: VibroGemmaForClassification, config: dict[str, Any]) -> None:
    from peft import LoraConfig, TaskType, get_peft_model

    lora = config["training"]["lora"]
    text_config = getattr(model.gemma.config, "text_config", None) or model.gemma.config
    num_layers = int(text_config.num_hidden_layers)
    last_n = min(int(lora.get("train_last_n_layers", num_layers)), num_layers)
    layers = list(range(num_layers - last_n, num_layers))
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora.get("dropout", 0.0)),
        target_modules=list(lora["target_modules"]),
        layers_to_transform=layers,
        layers_pattern="layers",
        bias="none",
    )
    model.gemma = get_peft_model(model.gemma, peft_config)
    lora_parameters = [
        name
        for name, parameter in model.gemma.named_parameters()
        if parameter.requires_grad and (".lora_A." in name or ".lora_B." in name)
    ]
    expected_fragments = tuple(
        f".language_model.layers.{layer_index}." for layer_index in layers
    )
    invalid = [
        name
        for name in lora_parameters
        if not any(fragment in name for fragment in expected_fragments)
    ]
    if not lora_parameters or invalid:
        raise RuntimeError(
            "Gemma 4 LoRA scope is empty or escaped the selected language-model layers: "
            f"count={len(lora_parameters)}, invalid={invalid[:10]}"
        )
    model.lora_scope_report = {
        "parameter_tensor_count": len(lora_parameters),
        "layers": layers,
        "all_language_model_scoped": True,
    }
    model.structured_generator.model = model.gemma


def run_alignment_training(
    config: dict[str, Any],
    *,
    resume: str | None = None,
    initialize_from: str | None = None,
) -> dict[str, Any]:
    return _run_lm_stage(
        config, stage="align", resume=resume, initialize_from=initialize_from
    )


def run_lora_training(
    config: dict[str, Any],
    *,
    resume: str | None = None,
    initialize_from: str | None = None,
) -> dict[str, Any]:
    return _run_lm_stage(
        config, stage="lora", resume=resume, initialize_from=initialize_from
    )


def _run_lm_stage(
    config: dict[str, Any],
    *,
    stage: Literal["align", "lora"],
    resume: str | None,
    initialize_from: str | None,
) -> dict[str, Any]:
    validate_training_controls(config)
    set_global_seed(int(config["project"].get("seed", 0)))
    training = config["training"]
    stage_cfg = training[stage]
    runtime = stage_training_settings(config, stage)
    gradient_accumulation_steps = int(runtime["gradient_accumulation_steps"])
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=str(training.get("mixed_precision", "no")),
    )
    if resume and initialize_from:
        raise ValueError("--resume and --initialize-from are mutually exclusive")
    model = VibroGemmaForClassification.from_config(config, training=True)
    initialization_source = resume or initialize_from
    if bool(stage_cfg.get("freeze_encoder", False)) and not initialization_source:
        raise ValueError(
            f"training.{stage}.freeze_encoder=true requires --resume or --initialize-from"
        )
    if initialization_source:
        validate_checkpoint_data_contract(config, initialization_source)
    source_stage = _checkpoint_stage(initialization_source)
    initialize_projector = bool(
        stage_cfg.get("initialize_projector_from_checkpoint", True)
    )
    if stage == "align":
        if initialization_source:
            _load_pretrain_encoder(
                model,
                initialization_source,
                initialize_projector=initialize_projector,
            )
        model.freeze_gemma()
    else:
        if initialization_source and source_stage != "lora":
            _load_pretrain_encoder(
                model,
                initialization_source,
                initialize_projector=initialize_projector,
            )
        _apply_lora(model, config)
        if initialization_source and source_stage == "lora":
            _load_lora_initialization(model, initialization_source)
    trainability_report = _configure_downstream_trainability(
        model,
        stage=stage,
        stage_cfg=stage_cfg,
    )
    lora_scope_report = getattr(model, "lora_scope_report", None)
    model.to(accelerator.device)
    _enforce_frozen_encoder_eval(model)

    parameter_groups = _gemma_optimizer_parameter_groups(model, stage_cfg)
    if not parameter_groups:
        raise RuntimeError(f"{stage} has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(stage_cfg.get("weight_decay", 0.0)),
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
    )
    raw_sampler = getattr(train_loader, "sampler", None)
    sampler_report = (
        raw_sampler.report()
        if hasattr(raw_sampler, "report")
        else {"strategy": "random_shuffle"}
    )
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
    state = TrainerState(stage=stage)
    # A checkpoint of the same stage restores optimizer/scheduler/RNG. A preceding
    # stage path was already used only to initialize modality components above.
    if resume and (Path(resume) / "trainer_state.json").exists():
        payload = json.loads((Path(resume) / "trainer_state.json").read_text(encoding="utf-8"))
        if payload.get("stage") == stage:
            state = load_checkpoint(
                resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                accelerator=accelerator,
            )
    checkpoint_root = resolve_from_config(config, config["project"]["checkpoint_dir"]) / stage
    log_every = max(1, int(stage_cfg.get("log_every_steps", training.get("log_every_steps", 10))))
    early_cfg = dict(stage_cfg.get("early_stopping") or {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = max(1, int(early_cfg.get("patience", 2)))
    early_min_delta = max(0.0, float(early_cfg.get("min_delta", 0.001)))
    early_minimum_epochs = max(1, int(early_cfg.get("minimum_epochs", 2)))
    early_stop_triggered = False
    structured_cfg = dict(training.get("structured_set_training") or {})
    guard_cfg = dict(structured_cfg.get("collapse_guard") or {})
    guard_enabled = bool(
        structured_cfg.get("enabled", False) and guard_cfg.get("enabled", True)
    )
    guard_warmup_epochs = max(0.0, float(guard_cfg.get("warmup_epochs", 0.0)))
    guard_warmup_steps = max(
        0,
        int(guard_cfg.get("warmup_steps", 100)),
        math.ceil(guard_warmup_epochs * optimizer_steps_per_epoch),
    )
    guard_check_every_steps = max(1, int(guard_cfg.get("check_every_steps", 25)))
    guard_window_examples = max(1, int(guard_cfg.get("window_examples", 128)))
    guard_failure_action = str(guard_cfg.get("failure_action", "raise"))
    if guard_failure_action not in {"raise", "warn"}:
        raise ValueError("collapse_guard.failure_action must be raise or warn")
    guard_counts = _empty_decision_counts()
    accelerator.print(
        json.dumps(
            {
                "stage": stage,
                "resume": resume,
                "initialize_from": initialize_from,
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
                "lora_scope": lora_scope_report,
                "train_vibration_components": bool(
                    stage_cfg.get("train_vibration_components", True)
                ),
                "initialize_projector_from_checkpoint": initialize_projector,
                "trainability": trainability_report,
                "early_stopping": {
                    "enabled": early_enabled,
                    "patience": early_patience,
                    "min_delta": early_min_delta,
                    "minimum_epochs": early_minimum_epochs,
                },
                "collapse_guard": {
                    "enabled": guard_enabled,
                    "warmup_steps": guard_warmup_steps,
                    "warmup_epochs": guard_warmup_epochs,
                    "check_every_steps": guard_check_every_steps,
                    "window_examples": guard_window_examples,
                    "failure_action": guard_failure_action,
                },
            },
            sort_keys=True,
        )
    )

    for epoch in range(state.epoch, epochs):
        set_loader_epoch(train_loader, epoch)
        model.train()
        _enforce_frozen_encoder_eval(accelerator.unwrap_model(model))
        epoch_started = time.perf_counter()
        interval_started = epoch_started
        interval_examples = 0
        interval_batches = 0
        last_loss = float("nan")
        last_structured_loss_components: dict[str, float] = {}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        accelerator.print(
            json.dumps(
                {
                    "stage": stage,
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
                output = model(prepared_batch)
                loss = output.loss
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite language-model loss at step {state.global_step}: {loss}")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        float(training.get("max_grad_norm", 1.0)),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if guard_enabled and state.global_step >= guard_warmup_steps:
                unwrapped_model = accelerator.unwrap_model(model)
                _accumulate_decision_stats(
                    accelerator,
                    guard_counts,
                    getattr(unwrapped_model, "last_decision_stats", None)
                    or getattr(output, "decision_stats", None),
                )
            if accelerator.sync_gradients:
                state.global_step += 1
                last_loss = float(loss.detach().float().item())
                unwrapped_model = accelerator.unwrap_model(model)
                raw_components = getattr(
                    unwrapped_model, "last_structured_loss_components", None
                ) or getattr(output, "structured_loss_components", None)
                if raw_components:
                    last_structured_loss_components = {
                        str(name): float(value.detach().float().item())
                        for name, value in raw_components.items()
                    }
                if (
                    guard_enabled
                    and state.global_step >= guard_warmup_steps
                    and state.global_step % guard_check_every_steps == 0
                    and guard_counts["example_count"] >= guard_window_examples
                ):
                    guard_metrics = _decision_guard_metrics(guard_counts)
                    failures = _guard_failures(guard_metrics, guard_cfg)
                    accelerator.print(
                        json.dumps(
                            {
                                "stage": stage,
                                "event": "structured_decision_guard",
                                "global_step": state.global_step,
                                "metrics": guard_metrics,
                                "failures": failures,
                            },
                            sort_keys=True,
                        )
                    )
                    guard_counts = _empty_decision_counts()
                    if failures and guard_failure_action == "raise":
                        raise RuntimeError(
                            "structured decision collapse guard failed: "
                            + "; ".join(failures)
                        )
                if state.global_step % log_every == 0:
                    now = time.perf_counter()
                    elapsed = max(now - interval_started, 1e-9)
                    progress = {
                        "stage": stage,
                        "event": "progress",
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "global_step": state.global_step,
                        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                        "loss": last_loss,
                        "learning_rate": [float(value) for value in scheduler.get_last_lr()],
                        "examples_per_second": interval_examples / elapsed,
                        "batches_per_second": interval_batches / elapsed,
                        "elapsed_epoch_minutes": (now - epoch_started) / 60.0,
                    }
                    if last_structured_loss_components:
                        progress["structured_loss_components"] = (
                            last_structured_loss_components
                        )
                    if torch.cuda.is_available():
                        progress["gpu_peak_memory_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
                    accelerator.print(json.dumps(progress, sort_keys=True))
                    interval_started = now
                    interval_examples = 0
                    interval_batches = 0

        validation_started = time.perf_counter()
        validation_loss = _evaluate_lm(accelerator, model, validation_loader)
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
            "structured_loss_components_last": last_structured_loss_components,
            "initialize_projector_from_checkpoint": initialize_projector,
            "trainability": trainability_report,
        }
        improved = (
            state.best_validation_loss is None
            or validation_loss < state.best_validation_loss - early_min_delta
        )
        if improved:
            state.best_validation_loss = validation_loss
            state.epochs_without_improvement = 0
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
        else:
            state.epochs_without_improvement += 1
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
                    "stage": stage,
                    "event": "epoch_end",
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "global_step": state.global_step,
                    "validation_loss": validation_loss,
                    "best_validation_loss": state.best_validation_loss,
                    "epoch_minutes": epoch_seconds / 60.0,
                    "validation_minutes": validation_seconds / 60.0,
                    "improved": improved,
                    "epochs_without_improvement": state.epochs_without_improvement,
                },
                sort_keys=True,
            )
        )
        if (
            early_enabled
            and state.epoch >= early_minimum_epochs
            and state.epochs_without_improvement >= early_patience
        ):
            state.completed = True
            state.stop_reason = (
                f"early_stopping: no validation improvement > {early_min_delta:g} "
                f"for {state.epochs_without_improvement} epoch(s)"
            )
            early_stop_triggered = True
            accelerator.print(
                json.dumps(
                    {
                        "stage": stage,
                        "event": "early_stop",
                        "epoch": state.epoch,
                        "best_validation_loss": state.best_validation_loss,
                        "reason": state.stop_reason,
                    },
                    sort_keys=True,
                )
            )
            break
    state.completed = True
    if not early_stop_triggered:
        state.stop_reason = "maximum_epochs_reached"
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
                "initialize_projector_from_checkpoint": initialize_projector,
                "trainability": trainability_report,
            },
        )
    return {
        "stage": stage,
        "global_step": state.global_step,
        "best_validation_loss": state.best_validation_loss,
        "stop_reason": state.stop_reason,
        "epochs_completed": state.epoch,
    }


@torch.inference_mode()
def _evaluate_lm(accelerator: Accelerator, model, loader) -> float:
    model.eval()
    losses: list[torch.Tensor] = []
    for prepared_batch in loader:
        output = model(prepared_batch)
        loss = output.loss
        if loss is None:
            raise RuntimeError("language-model validation returned no loss")
        losses.append(accelerator.gather_for_metrics(loss.detach().reshape(1)))
    if not losses:
        raise RuntimeError("validation loader produced no batches")
    return float(torch.cat(losses).mean().item())
