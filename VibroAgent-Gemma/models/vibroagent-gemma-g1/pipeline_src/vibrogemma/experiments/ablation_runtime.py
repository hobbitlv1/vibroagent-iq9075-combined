from __future__ import annotations

"""Runtime ablations that preserve the six-board, 84-token interface.

The module is intentionally independent of the training/evaluation entry points.
It modifies prepared numeric inputs and selected fusion modules through PyTorch
hooks.  The same mechanism is available in notebook evaluation and CLI
subprocesses through ``VIBROGEMMA_ABLATION_JSON``.

Frozen knockouts answer what a trained model currently relies on.  A causal
retraining run must activate the same specification from the beginning of
pretraining; the two forms of evidence must never be conflated.
"""

import contextlib
import contextvars
import inspect
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import nn

_ACTIVE_SPEC: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "vibrogemma_active_ablation", default=None
)
_PATCHED_CLASSES: dict[type[nn.Module], Any] = {}
_LAST_MANIFESTS: list[dict[str, Any]] = []

_INPUT_NAMES = (
    "modal_values",
    "modal_axis_mask",
    "modal_sample_mask",
    "wideband_features",
    "wideband_frequency_mask",
    "wideband_axis_mask",
    "physics_features",
    "physics_mask",
    "sensor_positions",
    "orientation_features",
    "quality_features",
)

# Physics layout used by the published 234-feature contract.
PHYSICS_FAMILIES: dict[str, tuple[int, ...]] = {
    "local": tuple(range(0, 81)),
    "reference_relative": tuple(range(81, 153)),
    "healthy_profile": tuple(range(153, 234)),
    "amplitude_statistics": tuple(
        idx
        for base in (0, 22, 44, 153, 175, 197)
        for idx in (base + 0, base + 1, base + 2, base + 3)
    ),
    "spectral_entropy": tuple(base + 4 for base in (0, 22, 44, 153, 175, 197)),
    "band_energy": tuple(
        idx
        for base in (0, 22, 44, 153, 175, 197)
        for idx in range(base + 5, base + 13)
    ),
    "dominant_peaks": tuple(
        idx
        for base in (0, 22, 44, 153, 175, 197)
        for idx in range(base + 13, base + 22)
    ),
    "cross_axis_covariance": tuple(range(66, 72)) + tuple(range(219, 225)),
    "modal_features": tuple(range(72, 81)) + tuple(range(225, 234)),
    "transmissibility": tuple(81 + axis * 24 + band * 3 for axis in range(3) for band in range(8)),
    "cross_power": tuple(82 + axis * 24 + band * 3 for axis in range(3) for band in range(8)),
    "coherence": tuple(83 + axis * 24 + band * 3 for axis in range(3) for band in range(8)),
}

WIDEBAND_FAMILIES: dict[str, tuple[int, ...]] = {
    "log_power": (0,),
    "local_phase": (1, 2),
    "reference_cross_magnitude": (3,),
    "reference_cross_phase": (4, 5),
    "reference_coherence": (6,),
    "cross_spectral_validity": (7,),
    "frequency_validity": (8,),
    "synchronized_relation": (3, 4, 5, 6),
}


def normalize_spec(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(spec or {})
    name = str(raw.get("name", "full"))
    raw["name"] = name

    aliases: dict[str, dict[str, Any]] = {
        "full": {},
        "no_modal": {"drop_branches": ["modal"]},
        "no_wideband": {"drop_branches": ["wideband"]},
        "no_physics": {"drop_branches": ["physics"]},
        "modal_only": {"drop_branches": ["wideband", "physics"]},
        "wideband_only": {"drop_branches": ["modal", "physics"]},
        "physics_only": {"drop_branches": ["modal", "wideband"]},
        "no_x": {"drop_axes": ["x"]},
        "no_y": {"drop_axes": ["y"]},
        "no_z": {"drop_axes": ["z"]},
        "x_only": {"drop_axes": ["y", "z"]},
        "y_only": {"drop_axes": ["x", "z"]},
        "z_only": {"drop_axes": ["x", "y"]},
        "xy": {"drop_axes": ["z"]},
        "xz": {"drop_axes": ["y"]},
        "yz": {"drop_axes": ["x"]},
        "no_local_phase": {"drop_wideband_families": ["local_phase"]},
        "no_cross_magnitude": {"drop_wideband_families": ["reference_cross_magnitude"]},
        "no_cross_phase": {"drop_wideband_families": ["reference_cross_phase"]},
        "no_coherence": {"drop_wideband_families": ["reference_coherence"]},
        "no_synchronized_relation": {"drop_wideband_families": ["synchronized_relation"]},
        "log_power_only": {"keep_wideband_channels": [0, 7, 8]},
        "no_local_physics": {"drop_physics_families": ["local"]},
        "no_reference_physics": {"drop_physics_families": ["reference_relative"]},
        "no_profile_physics": {"drop_physics_families": ["healthy_profile"]},
        "no_amplitude_statistics": {"drop_physics_families": ["amplitude_statistics"]},
        "no_band_energy": {"drop_physics_families": ["band_energy"]},
        "no_spectral_shape": {"drop_physics_families": ["spectral_entropy", "dominant_peaks"]},
        "no_modal_features": {"drop_physics_families": ["modal_features"]},
        "no_cross_axis": {"drop_physics_families": ["cross_axis_covariance"]},
        "no_transmissibility": {"drop_physics_families": ["transmissibility"]},
        "no_cross_power": {"drop_physics_families": ["cross_power"]},
        "no_physics_coherence": {"drop_physics_families": ["coherence"]},
        "no_modal_axis_fusion": {"disable_modules": ["modal_axis_fusion"]},
        "no_wideband_axis_fusion": {"disable_modules": ["wideband_axis_fusion"]},
        "no_physics_feature_fusion": {"disable_modules": ["physics_feature_fusion"]},
        "no_branch_fusion": {"disable_modules": ["branch_fusion"]},
        "no_reference_relation": {"disable_modules": ["reference_relation"]},
        "no_cross_board_fusion": {"disable_modules": ["cross_board_fusion"]},
        "no_position_metadata": {"drop_metadata": ["position"]},
        "no_orientation_metadata": {"drop_metadata": ["orientation"]},
        "no_quality_metadata": {"drop_metadata": ["quality"]},
        "no_axis_identity": {"disable_modules": ["axis_identity"]},
        "no_board_identity": {"disable_modules": ["board_identity"]},
        "no_branch_identity": {"disable_modules": ["branch_identity"]},
        "no_role_identity": {"disable_modules": ["role_identity"]},
        "no_token_position": {"disable_modules": ["token_position"]},
    }
    for key, value in aliases.get(name, {}).items():
        if key not in raw:
            raw[key] = value

    for key in (
        "drop_branches",
        "drop_axes",
        "drop_wideband_families",
        "drop_physics_families",
        "drop_metadata",
        "disable_modules",
    ):
        raw[key] = sorted({str(item) for item in raw.get(key, [])})
    if "keep_wideband_channels" in raw:
        raw["keep_wideband_channels"] = sorted({int(item) for item in raw["keep_wideband_channels"]})
    return raw


def _clone_zero_indices(value: torch.Tensor, *, dim: int, indices: Iterable[int]) -> torch.Tensor:
    indices = tuple(sorted(set(int(i) for i in indices if 0 <= int(i) < value.shape[dim])))
    if not indices:
        return value
    out = value.clone()
    selector = [slice(None)] * out.ndim
    selector[dim] = torch.tensor(indices, device=out.device, dtype=torch.long)
    out[tuple(selector)] = 0
    return out


def _clone_false_indices(value: torch.Tensor, *, dim: int, indices: Iterable[int]) -> torch.Tensor:
    indices = tuple(sorted(set(int(i) for i in indices if 0 <= int(i) < value.shape[dim])))
    if not indices:
        return value
    out = value.clone()
    selector = [slice(None)] * out.ndim
    selector[dim] = torch.tensor(indices, device=out.device, dtype=torch.long)
    out[tuple(selector)] = False if out.dtype == torch.bool else 0
    return out


def _mapping_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if kwargs:
        values = dict(kwargs)
        for index, item in enumerate(args):
            if index < len(_INPUT_NAMES) and _INPUT_NAMES[index] not in values:
                values[_INPUT_NAMES[index]] = item
        return values, True
    return {name: args[index] for index, name in enumerate(_INPUT_NAMES) if index < len(args)}, False


def _restore_call(values: dict[str, Any], args: tuple[Any, ...], kwargs_mode: bool):
    if kwargs_mode:
        new_args = list(args)
        new_kwargs = dict(values)
        for index in range(min(len(new_args), len(_INPUT_NAMES))):
            new_args[index] = values[_INPUT_NAMES[index]]
            new_kwargs.pop(_INPUT_NAMES[index], None)
        return tuple(new_args), new_kwargs
    new_args = list(args)
    for index, name in enumerate(_INPUT_NAMES):
        if index < len(new_args) and name in values:
            new_args[index] = values[name]
    return tuple(new_args), {}


def _input_pre_hook(spec: dict[str, Any]):
    axis_lookup = {"x": 0, "y": 1, "z": 2}
    dropped_axes = tuple(axis_lookup[item] for item in spec["drop_axes"] if item in axis_lookup)
    dropped_branches = set(spec["drop_branches"])
    metadata = set(spec["drop_metadata"])

    wideband_drop: set[int] = set()
    for family in spec["drop_wideband_families"]:
        wideband_drop.update(WIDEBAND_FAMILIES.get(family, ()))
    if "keep_wideband_channels" in spec:
        keep = set(spec["keep_wideband_channels"])
        wideband_drop.update(index for index in range(9) if index not in keep)

    physics_drop: set[int] = set()
    for family in spec["drop_physics_families"]:
        physics_drop.update(PHYSICS_FAMILIES.get(family, ()))

    def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        values, kwargs_mode = _mapping_from_call(args, kwargs)

        if dropped_axes:
            for name, dim in (("modal_values", 2), ("wideband_features", 2)):
                if isinstance(values.get(name), torch.Tensor):
                    values[name] = _clone_zero_indices(values[name], dim=dim, indices=dropped_axes)
            for name, dim in (("modal_axis_mask", 2), ("wideband_axis_mask", 2)):
                if isinstance(values.get(name), torch.Tensor):
                    values[name] = _clone_false_indices(values[name], dim=dim, indices=dropped_axes)

        if "modal" in dropped_branches:
            for name in ("modal_values",):
                if isinstance(values.get(name), torch.Tensor):
                    values[name] = torch.zeros_like(values[name])
            for name in ("modal_axis_mask", "modal_sample_mask"):
                if isinstance(values.get(name), torch.Tensor):
                    values[name] = torch.zeros_like(values[name])

        if "wideband" in dropped_branches:
            if isinstance(values.get("wideband_features"), torch.Tensor):
                values["wideband_features"] = torch.zeros_like(values["wideband_features"])
            for name in ("wideband_frequency_mask", "wideband_axis_mask"):
                if isinstance(values.get(name), torch.Tensor):
                    values[name] = torch.zeros_like(values[name])
        elif wideband_drop and isinstance(values.get("wideband_features"), torch.Tensor):
            values["wideband_features"] = _clone_zero_indices(
                values["wideband_features"], dim=3, indices=wideband_drop
            )

        if "physics" in dropped_branches:
            for name in ("physics_features", "physics_mask"):
                if isinstance(values.get(name), torch.Tensor):
                    values[name] = torch.zeros_like(values[name])
        elif physics_drop:
            if isinstance(values.get("physics_features"), torch.Tensor):
                values["physics_features"] = _clone_zero_indices(
                    values["physics_features"], dim=2, indices=physics_drop
                )
            if isinstance(values.get("physics_mask"), torch.Tensor):
                values["physics_mask"] = _clone_false_indices(
                    values["physics_mask"], dim=2, indices=physics_drop
                )

        if "position" in metadata and isinstance(values.get("sensor_positions"), torch.Tensor):
            values["sensor_positions"] = torch.zeros_like(values["sensor_positions"])
        if "orientation" in metadata and isinstance(values.get("orientation_features"), torch.Tensor):
            values["orientation_features"] = torch.zeros_like(values["orientation_features"])
        if "quality" in metadata and isinstance(values.get("quality_features"), torch.Tensor):
            values["quality_features"] = torch.zeros_like(values["quality_features"])

        return _restore_call(values, args, kwargs_mode)

    return hook


def _identity_hook(module: nn.Module, inputs: tuple[Any, ...], output: Any):
    if not inputs:
        return output
    candidate = inputs[0]
    if isinstance(output, torch.Tensor) and isinstance(candidate, torch.Tensor) and output.shape == candidate.shape:
        return candidate
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        if isinstance(candidate, torch.Tensor) and output[0].shape == candidate.shape:
            return (candidate, *output[1:])
    return output


def _zero_hook(module: nn.Module, inputs: tuple[Any, ...], output: Any):
    if isinstance(output, torch.Tensor):
        return torch.zeros_like(output)
    if isinstance(output, tuple):
        return tuple(torch.zeros_like(item) if isinstance(item, torch.Tensor) else item for item in output)
    return output


def _find_tokenizer(model: nn.Module) -> tuple[str, nn.Module]:
    for name, module in model.named_modules():
        try:
            params = inspect.signature(module.forward).parameters
        except (TypeError, ValueError):
            continue
        required = {"modal_values", "wideband_features", "physics_features"}
        if required.issubset(params):
            return name, module
    for attr in ("tokenizer", "signal_tokenizer", "vibration_tokenizer"):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module):
            return attr, module
    raise RuntimeError("Could not locate the multi-resolution vibration tokenizer")


def _module_target(kind: str, name: str) -> bool:
    lname = name.lower()
    patterns = {
        "modal_axis_fusion": ("modal", "axis_fusion"),
        "wideband_axis_fusion": ("wideband", "axis_fusion"),
        "physics_feature_fusion": ("feature_fusion",),
        "branch_fusion": ("branch_fusion",),
        "reference_relation": ("relation",),
        "cross_board_fusion": ("cross_board_fusion",),
        "axis_identity": ("axis_embedding",),
        "board_identity": ("board_embedding",),
        "branch_identity": ("branch_embedding",),
        "role_identity": ("role_embedding",),
        "token_position": ("token_position",),
    }
    return all(item in lname for item in patterns.get(kind, ("__never__",)))


def apply_ablation_to_model(model: nn.Module, spec: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_spec(spec)
    if normalized["name"] == "full" and not any(
        normalized.get(key)
        for key in (
            "drop_branches", "drop_axes", "drop_wideband_families",
            "drop_physics_families", "drop_metadata", "disable_modules"
        )
    ):
        manifest = {"spec": normalized, "tokenizer": None, "applied_modules": [], "missing_modules": []}
        setattr(model, "_vibrogemma_ablation_manifest", manifest)
        _LAST_MANIFESTS.append(manifest)
        return manifest

    tokenizer_name, tokenizer = _find_tokenizer(model)
    handles = [tokenizer.register_forward_pre_hook(_input_pre_hook(normalized), with_kwargs=True)]
    applied: list[dict[str, str]] = []
    missing: list[str] = []

    for kind in normalized["disable_modules"]:
        matches = [(name, module) for name, module in tokenizer.named_modules() if name and _module_target(kind, name)]
        if not matches:
            missing.append(kind)
            continue
        for name, module in matches:
            hook = _zero_hook if kind in {
                "reference_relation", "axis_identity", "board_identity", "branch_identity",
                "role_identity", "token_position"
            } else _identity_hook
            handles.append(module.register_forward_hook(hook))
            applied.append({"kind": kind, "module": f"{tokenizer_name}.{name}"})

    manifest = {
        "spec": normalized,
        "tokenizer": tokenizer_name,
        "applied_modules": applied,
        "missing_modules": missing,
        "handle_count": len(handles),
    }
    setattr(model, "_vibrogemma_ablation_handles", handles)
    setattr(model, "_vibrogemma_ablation_manifest", manifest)
    _LAST_MANIFESTS.append(manifest)
    return manifest


def consume_ablation_manifests() -> list[dict[str, Any]]:
    result = list(_LAST_MANIFESTS)
    _LAST_MANIFESTS.clear()
    return result


def _pipeline_classes() -> list[type[nn.Module]]:
    from vibrogemma.model import pipeline as pipeline_module

    candidates: list[type[nn.Module]] = []
    for name, value in vars(pipeline_module).items():
        if not isinstance(value, type):
            continue
        try:
            is_module = issubclass(value, nn.Module)
        except TypeError:
            continue
        if not is_module:
            continue
        lname = name.lower()
        if "pipeline" in lname or "vibrogemma" in lname:
            candidates.append(value)
    if not candidates:
        raise RuntimeError("No VibroGemma pipeline class was found for runtime ablation")
    return candidates


def install_constructor_patch() -> list[str]:
    patched: list[str] = []
    for cls in _pipeline_classes():
        if cls in _PATCHED_CLASSES:
            patched.append(cls.__name__)
            continue
        original = cls.__init__

        def wrapped(self, *args, __original=original, **kwargs):
            __original(self, *args, **kwargs)
            spec = _ACTIVE_SPEC.get()
            if spec is not None:
                apply_ablation_to_model(self, spec)

        cls.__init__ = wrapped  # type: ignore[method-assign]
        _PATCHED_CLASSES[cls] = original
        patched.append(cls.__name__)
    return patched


@contextlib.contextmanager
def ablation_context(spec: Mapping[str, Any] | None):
    install_constructor_patch()
    token = _ACTIVE_SPEC.set(normalize_spec(spec))
    try:
        yield
    finally:
        _ACTIVE_SPEC.reset(token)


def install_from_environment() -> dict[str, Any] | None:
    raw = os.environ.get("VIBROGEMMA_ABLATION_JSON")
    if not raw:
        return None
    spec = normalize_spec(json.loads(raw))
    install_constructor_patch()
    _ACTIVE_SPEC.set(spec)
    return spec
