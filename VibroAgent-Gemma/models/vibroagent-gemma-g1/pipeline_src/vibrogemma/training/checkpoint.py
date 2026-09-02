from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ..utils import atomic_write_json, utc_now_iso


@dataclass(slots=True)
class TrainerState:
    stage: str
    epoch: int = 0
    global_step: int = 0
    best_validation_loss: float | None = None
    epochs_without_improvement: int = 0
    completed: bool = False
    stop_reason: str | None = None


def localization_data_contract(config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the checkpoint-critical physical target layout, when configured."""

    if not config:
        return None
    protocol = dict(config.get("final_protocol") or {})
    panel = protocol.get("lumo_panel_levels")
    if panel is None:
        return None
    zone_contract = dict(protocol.get("lumo_zone_contract") or {})
    return {
        "schema": "vibrogemma-localization-data-contract-v1",
        "lumo_panel_levels": [str(value) for value in panel],
        "lumo_zone_contract_schema": zone_contract.get("schema"),
        "target_decision_semantics": str(
            config.get("training", {})
            .get("language_loss", {})
            .get("target_decision_semantics", "")
        ),
    }


def validate_checkpoint_data_contract(
    config: dict[str, Any] | None,
    directory: str | Path,
) -> None:
    """Reject an initialization checkpoint built for another physical board panel."""

    expected = localization_data_contract(config)
    if expected is None:
        return
    state_path = Path(directory) / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError(
            f"geometry-aligned training requires checkpoint contract metadata: {state_path}"
        )
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    observed = dict(payload.get("extra") or {}).get("localization_data_contract")
    if observed != expected:
        raise RuntimeError(
            "checkpoint localization data contract is missing or incompatible; "
            f"expected={expected}, observed={observed}. Retrain from fresh vibration components."
        )


def _unwrap(model: Any, accelerator: Any | None) -> Any:
    return accelerator.unwrap_model(model) if accelerator is not None else model


def _trainable_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    state = module.state_dict()
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
        if name in trainable_names or any(name.startswith(prefix + ".") for prefix in trainable_names)
    }


def save_checkpoint(
    directory: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: TrainerState,
    accelerator: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    unwrapped = _unwrap(model, accelerator)
    checkpoint_extra = dict(extra or {})
    data_contract = localization_data_contract(
        getattr(unwrapped, "config_payload", None)
    )
    if data_contract is not None:
        checkpoint_extra["localization_data_contract"] = data_contract
    if hasattr(unwrapped, "save_components"):
        unwrapped.save_components(destination, metadata={"stage": state.stage, "global_step": state.global_step})
    else:
        save_file(
            {name: tensor.detach().cpu().contiguous() for name, tensor in unwrapped.state_dict().items()},
            str(destination / "model.safetensors"),
        )
    gemma = getattr(unwrapped, "gemma", None)
    if gemma is not None:
        trainable = _trainable_state_dict(gemma)
        if trainable:
            save_file(trainable, str(destination / "gemma_trainable.safetensors"))
        if hasattr(gemma, "save_pretrained") and hasattr(gemma, "peft_config"):
            gemma.save_pretrained(destination / "adapter")
    torch.save(optimizer.state_dict(), destination / "optimizer.pt")
    torch.save(scheduler.state_dict() if scheduler is not None else {}, destination / "scheduler.pt")
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        destination / "rng.pt",
    )
    atomic_write_json(
        destination / "trainer_state.json",
        {**asdict(state), "saved_utc": utc_now_iso(), "extra": checkpoint_extra},
    )
    latest = destination.parent / "latest"
    temporary = destination.parent / ".latest.tmp"
    try:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(destination.name, target_is_directory=True)
        os.replace(temporary, latest)
    except OSError:
        atomic_write_json(destination.parent / "latest.json", {"path": str(destination.resolve())})


def load_checkpoint(
    directory: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    accelerator: Any | None = None,
) -> TrainerState:
    source = Path(directory).resolve()
    unwrapped = _unwrap(model, accelerator)
    validate_checkpoint_data_contract(
        getattr(unwrapped, "config_payload", None), source
    )
    if hasattr(unwrapped, "load_components") and (source / "vibration_components.safetensors").exists():
        unwrapped.load_components(source, strict=False)
    elif (source / "model.safetensors").exists():
        unwrapped.load_state_dict(load_file(str(source / "model.safetensors")), strict=False)
    if (source / "gemma_trainable.safetensors").exists() and hasattr(unwrapped, "gemma"):
        unwrapped.gemma.load_state_dict(load_file(str(source / "gemma_trainable.safetensors")), strict=False)
    if optimizer is not None and (source / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(source / "optimizer.pt", map_location="cpu", weights_only=False))
    if scheduler is not None and (source / "scheduler.pt").exists():
        scheduler.load_state_dict(torch.load(source / "scheduler.pt", map_location="cpu", weights_only=False))
    if (source / "rng.pt").exists():
        rng = torch.load(source / "rng.pt", map_location="cpu", weights_only=False)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
    payload = json.loads((source / "trainer_state.json").read_text(encoding="utf-8"))
    return TrainerState(
        stage=str(payload["stage"]),
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        best_validation_loss=payload.get("best_validation_loss"),
        epochs_without_improvement=int(payload.get("epochs_without_improvement", 0)),
        completed=bool(payload.get("completed", False)),
        stop_reason=payload.get("stop_reason"),
    )
