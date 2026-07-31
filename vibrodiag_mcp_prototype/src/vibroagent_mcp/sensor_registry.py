"""Sensor registry for building vibration monitoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import SensorConfig

VALID_AXES = {"norm", "magnitude", "vector_norm", "x", "y", "z", "0", "1", "2"}


class SensorRegistry:
    """Load and validate building vibration sensor configuration."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).expanduser().resolve()
        raw = _load_config(self.config_path)
        self.baseline_sensor_id = str(raw.get("baseline_sensor_id") or "baseline")
        sensors_raw = raw.get("sensors")
        if not isinstance(sensors_raw, list) or not sensors_raw:
            raise ValueError("Sensor registry config must contain a non-empty sensors list")

        self.sensors: dict[str, SensorConfig] = {}
        for item in sensors_raw:
            if not isinstance(item, dict):
                raise ValueError("Each sensor entry must be an object")
            config = _sensor_config_from_dict(item, config_path=self.config_path)
            if config.sensor_id in self.sensors:
                raise ValueError(f"Duplicate sensor_id in registry: {config.sensor_id}")
            self.sensors[config.sensor_id] = config

        if self.baseline_sensor_id not in self.sensors:
            raise ValueError(f"Baseline sensor {self.baseline_sensor_id!r} is not present in registry")
        if self.sensors[self.baseline_sensor_id].role != "baseline":
            baseline = self.sensors[self.baseline_sensor_id]
            self.sensors[self.baseline_sensor_id] = SensorConfig(
                **{**baseline.__dict__, "role": "baseline"}
            )
        self.aliases = _build_sensor_aliases(
            raw.get("aliases"),
            self.sensors,
            self.baseline_sensor_id,
        )

    def baseline(self, baseline_sensor_id: str | None = None) -> SensorConfig:
        selected = self.resolve_id(baseline_sensor_id or self.baseline_sensor_id)
        try:
            return self.sensors[selected]
        except KeyError as exc:
            raise ValueError(f"Baseline sensor {selected!r} is not present in registry") from exc

    def targets(self, sensor_ids: list[str] | tuple[str, ...] | None = None, baseline_sensor_id: str | None = None) -> list[SensorConfig]:
        baseline_id = self.resolve_id(baseline_sensor_id or self.baseline_sensor_id)
        if sensor_ids is None:
            return [
                sensor
                for sensor_id, sensor in self.sensors.items()
                if sensor_id != baseline_id
            ]
        resolved = self.resolve_selected(sensor_ids)
        return [sensor for sensor in resolved if sensor.sensor_id != baseline_id]

    def resolve_selected(self, sensor_ids: list[str] | tuple[str, ...]) -> list[SensorConfig]:
        resolved = []
        seen: set[str] = set()
        for sensor_id in sensor_ids:
            canonical = self.resolve_id(sensor_id)
            if canonical not in self.sensors:
                raise ValueError(
                    f"Sensor {sensor_id!r} is not present in registry. "
                    f"Available sensor IDs: {', '.join(self.sensors)}. "
                    f"Accepted aliases: {', '.join(sorted(self.aliases)) or 'none'}."
                )
            if canonical in seen:
                continue
            seen.add(canonical)
            resolved.append(self.sensors[canonical])
        return resolved

    def resolve_id(self, sensor_id: str) -> str:
        selected = str(sensor_id or "").strip()
        return self.aliases.get(selected, selected)

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "baseline_sensor_id": self.baseline_sensor_id,
            "sensors": [sensor.__dict__ for sensor in self.sensors.values()],
            "aliases": dict(self.aliases),
        }


def load_sensor_registry(config_path: str | Path) -> SensorRegistry:
    return SensorRegistry(config_path)


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Sensor registry config does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = _load_yaml_or_simple(text)
    if not isinstance(data, dict):
        raise ValueError("Sensor registry config must decode to an object")
    return data


def _load_yaml_or_simple(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _parse_simple_yaml(text)
    data = yaml.safe_load(text)
    return data or {}


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the simple YAML subset used by config/sensors.example.yaml.

    This is intentionally small: top-level scalars plus a list of sensor objects.
    If users need richer YAML, installing PyYAML enables full safe_load parsing.
    """
    data: dict[str, Any] = {}
    current_list_key: str | None = None
    current_item: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        if indent == 0 and stripped.endswith(":"):
            current_list_key = stripped[:-1]
            data[current_list_key] = []
            current_item = None
            continue

        if indent == 0 and ":" in stripped:
            key, value = stripped.split(":", 1)
            data[key.strip()] = _parse_scalar(value.strip())
            current_list_key = None
            current_item = None
            continue

        if current_list_key and stripped.startswith("- "):
            current_item = {}
            data[current_list_key].append(current_item)
            rest = stripped[2:].strip()
            if rest and ":" in rest:
                key, value = rest.split(":", 1)
                current_item[key.strip()] = _parse_scalar(value.strip())
            continue

        if current_item is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current_item[key.strip()] = _parse_scalar(value.strip())
            continue

        raise ValueError(f"Unsupported YAML line in sensor registry: {raw_line!r}")

    return data


def _parse_scalar(value: str) -> Any:
    if value == "":
        return None
    unquoted = value.strip()
    if (unquoted.startswith('"') and unquoted.endswith('"')) or (unquoted.startswith("'") and unquoted.endswith("'")):
        return unquoted[1:-1]
    lower = unquoted.lower()
    if lower in {"null", "none"}:
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(unquoted)
    except ValueError:
        pass
    try:
        return float(unquoted)
    except ValueError:
        return unquoted


def _sensor_config_from_dict(item: dict[str, Any], *, config_path: Path) -> SensorConfig:
    sensor_id = str(item.get("sensor_id") or "").strip()
    if not sensor_id:
        raise ValueError("Each sensor must define sensor_id")
    role = str(item.get("role") or "target").strip().lower()
    if role not in {"baseline", "target"}:
        raise ValueError(f"Sensor {sensor_id!r} role must be baseline or target")
    acquisition_folder = _resolve_optional_path(item.get("acquisition_folder"), config_path=config_path, must_be_dir=True)
    replay_signal_path = _resolve_optional_path(item.get("replay_signal_path"), config_path=config_path, must_be_dir=False)
    sampling_rate = item.get("sampling_rate_hz")
    return SensorConfig(
        sensor_id=sensor_id,
        location=str(item.get("location") or sensor_id),
        role=role,  # type: ignore[arg-type]
        acquisition_folder=acquisition_folder,
        hsd_sensor_name=str(item.get("hsd_sensor_name") or "iis3dwb_acc"),
        axis=_normalize_axis(item.get("axis") or "z"),
        replay_signal_path=replay_signal_path,
        sampling_rate_hz=_normalize_sampling_rate(sampling_rate, sensor_id=sensor_id),
    )


def _build_sensor_aliases(
    raw_aliases: Any,
    sensors: dict[str, SensorConfig],
    baseline_sensor_id: str,
) -> dict[str, str]:
    aliases: dict[str, str] = {}

    if isinstance(raw_aliases, dict):
        for alias, target in raw_aliases.items():
            alias_text = str(alias).strip()
            target_text = str(target).strip()
            if alias_text and target_text in sensors:
                aliases[alias_text] = target_text

    if baseline_sensor_id in sensors:
        aliases.setdefault("reference", baseline_sensor_id)
        aliases.setdefault("baseline_sensor", baseline_sensor_id)

    for sensor_id in sensors:
        if sensor_id.startswith("target_"):
            suffix = sensor_id.removeprefix("target_")
            if suffix.isdigit():
                aliases.setdefault(f"sensor_{suffix}", sensor_id)
                aliases.setdefault(f"board_{suffix}", sensor_id)
                aliases.setdefault(f"target_{int(suffix):02d}", sensor_id)
                aliases.setdefault(f"sensor_{int(suffix):02d}", sensor_id)
                aliases.setdefault(f"board_{int(suffix):02d}", sensor_id)

    return aliases


def _resolve_optional_path(value: Any, *, config_path: Path, must_be_dir: bool) -> str | None:
    if value in (None, ""):
        return None
    raw = Path(str(value)).expanduser()
    candidates = _path_candidates(raw, config_path=config_path)
    for candidate in candidates:
        if candidate.exists() and (candidate.is_dir() if must_be_dir else candidate.is_file()):
            return str(candidate.resolve())
    return str(candidates[0].resolve())


def _path_candidates(raw: Path, *, config_path: Path) -> list[Path]:
    remapped = _stdatalog_examples_remap_candidates(raw, config_path=config_path)
    if raw.is_absolute():
        candidates = [*remapped, raw] if remapped else [raw]
    else:
        candidates = [config_path.parent / raw, Path.cwd() / raw, *remapped]
    return _dedupe_paths(candidates)


def _stdatalog_examples_remap_candidates(raw: Path, *, config_path: Path) -> list[Path]:
    """Remap legacy absolute stdatalog_examples paths into this checkout.

    The live registry shipped with some deployments used absolute paths from the
    original EVK home directory. When the project is moved or unzipped elsewhere,
    those paths make the graph attempt to read folders that do not exist. Keep the
    configured value as the first candidate, but also try the same suffix under
    any ancestor that contains a sibling stdatalog_examples directory.
    """
    parts = raw.parts
    if "stdatalog_examples" not in parts:
        return []
    marker_index = parts.index("stdatalog_examples")
    suffix = Path(*parts[marker_index + 1 :]) if marker_index + 1 < len(parts) else Path()
    roots = [config_path.parent, *config_path.parents, Path.cwd(), *Path.cwd().parents]
    return [root / "stdatalog_examples" / suffix for root in roots]


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.expanduser().resolve(strict=False))
        except RuntimeError:
            key = str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _normalize_axis(value: Any) -> str:
    axis = str(value or "z").strip().lower()
    if axis not in VALID_AXES:
        raise ValueError("axis must be one of: norm, magnitude, vector_norm, x, y, z, 0, 1, 2")
    return axis


def _normalize_sampling_rate(value: Any, *, sensor_id: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        sampling_rate = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Sensor {sensor_id!r} sampling_rate_hz must be numeric") from None
    if sampling_rate <= 0:
        raise ValueError(f"Sensor {sensor_id!r} sampling_rate_hz must be positive")
    if not sampling_rate.is_integer():
        raise ValueError(f"Sensor {sensor_id!r} sampling_rate_hz must be a whole-number Hz value")
    return int(sampling_rate)
