from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise ConfigurationError(f"Configuration file does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Top-level YAML must be a mapping: {source}")
    return _expand(payload)


def load_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    config["_config_path"] = str(Path(path).resolve())
    config["_config_dir"] = str(Path(path).resolve().parent)
    validate_top_level(config)
    return config


def validate_top_level(config: dict[str, Any]) -> None:
    if "project" not in config:
        raise ConfigurationError("Missing project configuration")
    if "model" not in config:
        raise ConfigurationError("Missing model configuration")
    validate_training_controls(config)


def _string_list(value: Any, *, name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{name} must be a non-empty list when configured")
    if any(not isinstance(item, str) for item in value):
        raise ConfigurationError(f"{name} must contain only strings")
    result = [str(item).strip() for item in value]
    if any(not item for item in result):
        raise ConfigurationError(f"{name} cannot contain empty values")
    if len(result) != len(set(result)):
        raise ConfigurationError(f"{name} cannot contain duplicates")
    return result


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ConfigurationError(f"{name} must be finite")
    return result


def validate_training_controls(config: dict[str, Any]) -> None:
    """Validate opt-in corpus, representation, and encoder-freeze controls.

    Every new control has a missing-key default matching the historical B0
    behaviour. This validator therefore accepts legacy configurations unchanged.
    """

    training = config.get("training")
    if training is None:
        return
    if not isinstance(training, dict):
        raise ConfigurationError("training must be a mapping")
    pretrain = training.get("pretrain") or {}
    if not isinstance(pretrain, dict):
        raise ConfigurationError("training.pretrain must be a mapping")

    consistency_weight = _finite_float(
        pretrain.get("final_token_consistency_weight", 0.0),
        name="training.pretrain.final_token_consistency_weight",
    )
    if consistency_weight < 0.0:
        raise ConfigurationError(
            "training.pretrain.final_token_consistency_weight must be finite and non-negative"
        )
    branch_weights = [
        _finite_float(
            pretrain.get("modal_loss_weight", 1.0),
            name="training.pretrain.modal_loss_weight",
        ),
        _finite_float(
            pretrain.get("wideband_loss_weight", 1.0),
            name="training.pretrain.wideband_loss_weight",
        ),
        _finite_float(
            pretrain.get("physics_loss_weight", 0.5),
            name="training.pretrain.physics_loss_weight",
        ),
    ]
    if any(value < 0.0 for value in branch_weights):
        raise ConfigurationError("pretraining branch loss weights must be finite and non-negative")
    if consistency_weight > 0.0 and sum(branch_weights) <= 0.0:
        raise ConfigurationError(
            "final-token consistency requires a positive branch reconstruction anchor"
        )
    reconstruction_source = str(
        pretrain.get("reconstruction_token_source", "branch")
    )
    if reconstruction_source not in {"branch", "fused"}:
        raise ConfigurationError(
            "training.pretrain.reconstruction_token_source must be 'branch' or 'fused'"
        )

    corpus = pretrain.get("corpus") or {}
    if not isinstance(corpus, dict):
        raise ConfigurationError("training.pretrain.corpus must be a mapping")
    allowed_corpus_keys = {
        "include_dataset_ids",
        "include_training_roles",
        "train_only_dataset_ids",
    }
    unknown_corpus_keys = set(corpus) - allowed_corpus_keys
    if unknown_corpus_keys:
        raise ConfigurationError(
            "unsupported training.pretrain.corpus keys: "
            + ", ".join(sorted(unknown_corpus_keys))
        )
    include_datasets = _string_list(
        corpus.get("include_dataset_ids"),
        name="training.pretrain.corpus.include_dataset_ids",
    )
    roles = _string_list(
        corpus.get("include_training_roles"),
        name="training.pretrain.corpus.include_training_roles",
    )
    train_only = _string_list(
        corpus.get("train_only_dataset_ids"),
        name="training.pretrain.corpus.train_only_dataset_ids",
    )
    if roles is not None:
        allowed_roles = {"supervised", "representation_only"}
        unknown_roles = set(roles) - allowed_roles
        if unknown_roles:
            raise ConfigurationError(
                "unsupported encoder pretraining role(s): "
                + ", ".join(sorted(unknown_roles))
            )
    if include_datasets is not None and train_only is not None:
        outside = set(train_only) - set(include_datasets)
        if outside:
            raise ConfigurationError(
                "train_only_dataset_ids must be included in include_dataset_ids: "
                + ", ".join(sorted(outside))
            )

    sampling = pretrain.get("sampling") or {}
    if not isinstance(sampling, dict):
        raise ConfigurationError("training.pretrain.sampling must be a mapping")
    strategy = str(sampling.get("strategy", "random"))
    allowed_pretrain_strategies = {
        "random",
        "shuffle",
        "dataset_complete_group_window_uniform",
    }
    if strategy not in allowed_pretrain_strategies:
        raise ConfigurationError(
            f"unsupported encoder pretraining sampling strategy: {strategy}"
        )
    samples_per_epoch = sampling.get("samples_per_epoch")
    if samples_per_epoch is not None:
        if isinstance(samples_per_epoch, bool) or not isinstance(
            samples_per_epoch, int
        ):
            raise ConfigurationError(
                "training.pretrain.sampling.samples_per_epoch must be an integer >= 1"
            )
        if samples_per_epoch < 1:
            raise ConfigurationError(
                "training.pretrain.sampling.samples_per_epoch must be an integer >= 1"
            )

    for stage_name in ("align", "lora"):
        stage = training.get(stage_name) or {}
        if not isinstance(stage, dict):
            raise ConfigurationError(f"training.{stage_name} must be a mapping")
        stage_corpus = stage.get("corpus") or {}
        if not isinstance(stage_corpus, dict):
            raise ConfigurationError(
                f"training.{stage_name}.corpus must be a mapping"
            )
        unknown_stage_corpus_keys = set(stage_corpus) - {"include_dataset_ids"}
        if unknown_stage_corpus_keys:
            raise ConfigurationError(
                f"unsupported training.{stage_name}.corpus keys: "
                + ", ".join(sorted(unknown_stage_corpus_keys))
            )
        _string_list(
            stage_corpus.get("include_dataset_ids"),
            name=f"training.{stage_name}.corpus.include_dataset_ids",
        )
        for key in ("freeze_encoder", "initialize_projector_from_checkpoint"):
            if key in stage and not isinstance(stage[key], bool):
                raise ConfigurationError(f"training.{stage_name}.{key} must be boolean")
        if "train_vibration_components" in stage and not isinstance(
            stage["train_vibration_components"], bool
        ):
            raise ConfigurationError(
                f"training.{stage_name}.train_vibration_components must be boolean"
            )


def require(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ConfigurationError(f"Missing required configuration key: {dotted_key}")
        value = value[part]
    return value


def get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def config_project_root(config: dict[str, Any]) -> Path:
    """Return the project root for both flat and deeply nested config layouts.

    Historical configurations lived directly under ``<root>/configs``. Recovery
    notebooks store immutable configs more deeply, for example
    ``<root>/configs/<run>/r3/seed.yaml``. The previous resolver only recognized
    the flat form and therefore treated the ``r3`` directory as the project root.
    """

    config_path = Path(config.get("_config_path", ".")).resolve()
    config_dir = config_path.parent
    for ancestor in (config_dir, *config_dir.parents):
        if ancestor.name == "configs":
            return ancestor.parent
    return config_dir


def resolve_from_config(config: dict[str, Any], path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (config_project_root(config) / candidate).resolve()


def resolve_static_asset(config: dict[str, Any], path: str | Path) -> Path:
    """Resolve a repository/static asset without depending on config nesting.

    Runtime projects may copy static assets beside their ``configs`` directory.
    Editable/source installs also retain the assets in the repository tree. We
    accept either location and fail with the attempted paths when neither exists.
    """

    candidate = Path(path)
    if candidate.is_absolute():
        if candidate.is_file():
            return candidate
        raise ConfigurationError(f"Static asset does not exist: {candidate}")

    attempted: list[Path] = []
    configured = resolve_from_config(config, candidate)
    attempted.append(configured)

    env_root = os.environ.get("VIBROGEMMA_ASSET_ROOT", "").strip()
    if env_root:
        attempted.append((Path(env_root).expanduser().resolve() / candidate).resolve())

    # config.py is <repository>/src/vibrogemma/config.py in editable/source runs.
    repository_root = Path(__file__).resolve().parents[2]
    attempted.append((repository_root / candidate).resolve())

    for resolved in attempted:
        if resolved.is_file():
            return resolved
    joined = ", ".join(str(value) for value in attempted)
    raise ConfigurationError(
        f"Static asset {candidate} was not found; attempted: {joined}"
    )
