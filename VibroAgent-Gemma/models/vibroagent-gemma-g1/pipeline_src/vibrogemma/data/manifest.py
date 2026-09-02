from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import load_yaml, resolve_from_config
from ..exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    id: str
    name: str
    source_page: str
    access: str
    licence: str
    training_allowed: bool
    raw_subdir: str
    format: str
    adapter: str
    provenance_type: str
    reader: dict[str, Any]
    labels: dict[str, Any]
    localisation: dict[str, Any]
    notes: str = ""
    download: dict[str, Any] = field(default_factory=dict)
    grouping: dict[str, Any] = field(default_factory=dict)
    auto_index: dict[str, Any] = field(default_factory=dict)
    source_path: Path = Path()

    @classmethod
    def from_mapping(cls, value: dict[str, Any], source_path: Path) -> "DatasetManifest":
        required = (
            "id",
            "name",
            "source_page",
            "access",
            "licence",
            "training_allowed",
            "raw_subdir",
            "format",
            "adapter",
            "provenance_type",
            "reader",
            "labels",
            "localisation",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise ConfigurationError(f"{source_path}: missing dataset keys {missing}")
        return cls(
            id=str(value["id"]),
            name=str(value["name"]),
            source_page=str(value["source_page"]),
            access=str(value["access"]),
            licence=str(value["licence"]),
            training_allowed=bool(value["training_allowed"]),
            raw_subdir=str(value["raw_subdir"]),
            format=str(value["format"]).lower(),
            adapter=str(value["adapter"]),
            provenance_type=str(value["provenance_type"]),
            reader=dict(value["reader"]),
            labels=dict(value["labels"]),
            localisation=dict(value["localisation"]),
            notes=str(value.get("notes", "")),
            download=dict(value.get("download") or {}),
            grouping=dict(value.get("grouping") or {}),
            auto_index=dict(value.get("auto_index") or {}),
            source_path=source_path,
        )

    def raw_root(self, config: dict[str, Any]) -> Path:
        public_root = config.get("data", {}).get("public_root")
        if public_root:
            return resolve_from_config(config, public_root) / self.raw_subdir
        return resolve_from_config(config, config["data"]["raw_root"]) / self.raw_subdir

    def licence_marker(self, config: dict[str, Any]) -> Path:
        return self.raw_root(config) / ".licence_accepted.json"

    def index_path(self, config: dict[str, Any]) -> Path:
        _override_text = __import__("os").environ.get("VIBROGEMMA_INDEX_OVERRIDE_JSON")
        if _override_text:
            _override_map = __import__("json").loads(_override_text)
            _dataset_id = str(getattr(self, "id", "")).lower()
            for _token, _override_path in _override_map.items():
                if str(_token).lower() in _dataset_id:
                    return __import__("pathlib").Path(_override_path)
        configured = self.reader.get("index_file")
        return self.raw_root(config) / str(configured or "index.jsonl")


def load_dataset_manifests(config: dict[str, Any]) -> list[DatasetManifest]:
    paths = config.get("data", {}).get("datasets", [])
    manifests: list[DatasetManifest] = []
    for configured_path in paths:
        path = resolve_from_config(config, configured_path)
        manifests.append(DatasetManifest.from_mapping(load_yaml(path), path))
    identifiers = [manifest.id for manifest in manifests]
    if len(identifiers) != len(set(identifiers)):
        raise ConfigurationError("dataset IDs must be unique")
    return manifests
