from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import yaml

from ..exceptions import DataContractError, ProvenanceError
from ..utils import atomic_write_json, sha256_file, utc_now_iso

_DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
_DRIVE_FILES_ENDPOINT = "https://www.googleapis.com/drive/v3/files"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _identifier_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PrivateBundleFile:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PrivateBundleDataset:
    dataset_id: str
    target_subdir: str
    source_prefix: str
    source_bytes: int
    archive_bytes: int
    files: tuple[PrivateBundleFile, ...]
    final_archive: str
    include_patterns: tuple[str, ...]
    nested_archives: tuple[str, ...]
    access_gate: str
    redistribution_forbidden: bool
    training_eligible: bool
    ineligibility_reason: str | None

    @property
    def file_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.files)


@dataclass(frozen=True, slots=True)
class PrivateDriveBundle:
    schema_version: str
    bundle_name: str
    datasets: Mapping[str, PrivateBundleDataset]


@dataclass(frozen=True, slots=True)
class DriveFileMetadata:
    file_id: str
    name: str
    size: int
    mime_type: str
    modified_time: str | None
    md5_checksum: str | None


def _safe_relative_path(value: str, *, field: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DataContractError(f"{field} must be a safe relative path, got {value!r}")
    return normalized


def load_private_drive_bundle(path: str | Path) -> PrivateDriveBundle:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataContractError(f"private Drive bundle manifest must be a mapping: {source}")
    schema = str(payload.get("schema_version", ""))
    if schema != "vibrogemma.private-drive-bundle.v1":
        raise DataContractError(f"unsupported private Drive bundle schema {schema!r}")
    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, dict) or not raw_datasets:
        raise DataContractError("private Drive bundle manifest has no datasets")

    datasets: dict[str, PrivateBundleDataset] = {}
    seen_files: set[str] = set()
    for dataset_id, raw in raw_datasets.items():
        if not isinstance(raw, dict):
            raise DataContractError(f"dataset {dataset_id!r} bundle definition must be a mapping")
        files: list[PrivateBundleFile] = []
        for item in raw.get("files") or []:
            if not isinstance(item, dict):
                raise DataContractError(f"dataset {dataset_id}: file definition must be a mapping")
            name = Path(str(item.get("name", ""))).name
            if name != str(item.get("name", "")) or not name:
                raise DataContractError(f"dataset {dataset_id}: bundle file name must be a basename")
            if name in seen_files:
                raise DataContractError(f"duplicate bundle file name across datasets: {name}")
            seen_files.add(name)
            size = int(item.get("size", -1))
            checksum = str(item.get("sha256", "")).lower()
            if size <= 0 or _SHA256_PATTERN.fullmatch(checksum) is None:
                raise DataContractError(f"dataset {dataset_id}: invalid size/checksum for {name}")
            files.append(PrivateBundleFile(name=name, size=size, sha256=checksum))
        if not files:
            raise DataContractError(f"dataset {dataset_id}: no bundle files")
        final_archive = Path(str(raw.get("final_archive", ""))).name
        if final_archive not in {item.name for item in files}:
            raise DataContractError(f"dataset {dataset_id}: final_archive is not in files")
        source_prefix = _safe_relative_path(str(raw.get("source_prefix", "")), field="source_prefix")
        target_subdir = _safe_relative_path(str(raw.get("target_subdir", "")), field="target_subdir")
        patterns = tuple(
            _safe_relative_path(str(item), field="include_patterns") for item in raw.get("include_patterns") or []
        )
        if not patterns:
            raise DataContractError(f"dataset {dataset_id}: no selective extraction patterns")
        nested = tuple(
            _safe_relative_path(str(item), field="nested_archives") for item in raw.get("nested_archives") or []
        )
        dataset = PrivateBundleDataset(
            dataset_id=str(dataset_id),
            target_subdir=target_subdir,
            source_prefix=source_prefix,
            source_bytes=int(raw.get("source_bytes", 0)),
            archive_bytes=int(raw.get("archive_bytes", sum(item.size for item in files))),
            files=tuple(files),
            final_archive=final_archive,
            include_patterns=patterns,
            nested_archives=nested,
            access_gate=str(raw.get("access_gate", "open_licence")),
            redistribution_forbidden=bool(raw.get("redistribution_forbidden", False)),
            training_eligible=bool(raw.get("training_eligible", True)),
            ineligibility_reason=(
                str(raw["ineligibility_reason"]) if raw.get("ineligibility_reason") is not None else None
            ),
        )
        if dataset.source_bytes <= 0 or dataset.archive_bytes <= 0:
            raise DataContractError(f"dataset {dataset_id}: source/archive byte counts must be positive")
        if sum(item.size for item in dataset.files) != dataset.archive_bytes:
            raise DataContractError(
                f"dataset {dataset_id}: declared archive_bytes does not equal the sum of bundle parts"
            )
        datasets[dataset.dataset_id] = dataset
    return PrivateDriveBundle(
        schema_version=schema,
        bundle_name=str(payload.get("bundle_name", "private Drive dataset bundle")),
        datasets=datasets,
    )


class GoogleDriveReadOnlyClient:
    """Minimal Google Drive v3 client with resumable authenticated media reads."""

    def __init__(self, authorized_session: Any) -> None:
        self.session = authorized_session

    @classmethod
    def from_default_credentials(cls) -> "GoogleDriveReadOnlyClient":
        try:
            import google.auth
            from google.auth.transport.requests import AuthorizedSession
        except ImportError as exc:  # pragma: no cover - exercised in Colab/runtime environments.
            raise RuntimeError(
                "Google Drive support requires google-auth. Install the project 'drive' extra or google-auth."
            ) from exc
        credentials, _ = google.auth.default(scopes=[_DRIVE_READONLY_SCOPE])
        return cls(AuthorizedSession(credentials))

    def list_folder(self, folder_id: str) -> dict[str, DriveFileMetadata]:
        if not folder_id or "/" in folder_id:
            raise DataContractError("folder_id must be a raw Google Drive folder ID")
        files: dict[str, DriveFileMetadata] = {}
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "q": f"'{folder_id}' in parents and trashed = false",
                "pageSize": 1000,
                "spaces": "drive",
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
                "fields": "nextPageToken,files(id,name,size,mimeType,modifiedTime,md5Checksum)",
            }
            if page_token:
                params["pageToken"] = page_token
            response = self.session.get(_DRIVE_FILES_ENDPOINT, params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()
            for item in payload.get("files") or []:
                name = str(item["name"])
                if name in files:
                    raise ProvenanceError(f"private Drive folder contains duplicate direct-child name {name!r}")
                size_raw = item.get("size")
                if size_raw is None:
                    continue
                files[name] = DriveFileMetadata(
                    file_id=str(item["id"]),
                    name=name,
                    size=int(size_raw),
                    mime_type=str(item.get("mimeType", "")),
                    modified_time=str(item.get("modifiedTime")) if item.get("modifiedTime") else None,
                    md5_checksum=str(item.get("md5Checksum")) if item.get("md5Checksum") else None,
                )
            page_token = payload.get("nextPageToken")
            if not page_token:
                break
        return files

    def download_resumable(
        self,
        metadata: DriveFileMetadata,
        destination: str | Path,
        *,
        expected_size: int,
        expected_sha256: str,
        chunk_bytes: int = 16 * 1024 * 1024,
        retries: int = 8,
    ) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if metadata.size != expected_size:
            raise ProvenanceError(
                f"Drive size mismatch for {metadata.name}: expected {expected_size}, found {metadata.size}"
            )
        if destination.exists():
            if destination.stat().st_size == expected_size and sha256_file(destination) == expected_sha256:
                print(f"[drive] verified cached {metadata.name}")
                return destination
            destination.unlink()
        partial = destination.with_name(destination.name + ".part")
        if partial.exists() and partial.stat().st_size > expected_size:
            partial.unlink()

        media_url = f"{_DRIVE_FILES_ENDPOINT}/{metadata.file_id}"
        attempt = 0
        last_reported = -1
        while (partial.stat().st_size if partial.exists() else 0) < expected_size:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            params = {"alt": "media", "supportsAllDrives": "true"}
            try:
                with self.session.get(
                    media_url,
                    params=params,
                    headers=headers,
                    stream=True,
                    timeout=(60, 600),
                ) as response:
                    if offset and response.status_code == 200:
                        partial.unlink(missing_ok=True)
                        offset = 0
                    elif offset and response.status_code != 206:
                        response.raise_for_status()
                    else:
                        response.raise_for_status()
                    mode = "ab" if offset else "wb"
                    with partial.open(mode) as handle:
                        for block in response.iter_content(chunk_size=chunk_bytes):
                            if not block:
                                continue
                            handle.write(block)
                            current = handle.tell()
                            progress_bucket = current // (256 * 1024 * 1024)
                            if progress_bucket != last_reported:
                                last_reported = progress_bucket
                                print(
                                    f"[drive] {metadata.name}: {current / (1024**3):.2f}/"
                                    f"{expected_size / (1024**3):.2f} GiB"
                                )
                        handle.flush()
                        os.fsync(handle.fileno())
                attempt = 0
            except Exception as exc:
                attempt += 1
                if attempt > retries:
                    raise RuntimeError(
                        f"Drive download failed after {retries} retries for {metadata.name}; "
                        f"partial file retained at {partial}"
                    ) from exc
                delay = min(60.0, 2.0**attempt)
                print(f"[drive] retry {attempt}/{retries} for {metadata.name} after {delay:.0f}s: {exc}")
                time.sleep(delay)

        actual_size = partial.stat().st_size
        if actual_size != expected_size:
            raise ProvenanceError(
                f"downloaded size mismatch for {metadata.name}: expected {expected_size}, got {actual_size}"
            )
        actual_sha = sha256_file(partial)
        if actual_sha != expected_sha256:
            partial.unlink(missing_ok=True)
            raise ProvenanceError(
                f"SHA-256 mismatch for {metadata.name}: expected {expected_sha256}, got {actual_sha}"
            )
        partial.replace(destination)
        print(f"[drive] checksum verified: {metadata.name}")
        return destination


def _find_7zip() -> str:
    for executable in ("7zz", "7z"):
        found = shutil.which(executable)
        if found:
            return found
    raise RuntimeError("7-Zip is required for split ZIP extraction; install p7zip-full (or 7zip)")


def _list_7zip_paths(archive: Path, executable: str) -> list[str]:
    process = subprocess.run(
        [executable, "l", "-slt", str(archive)],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    paths: list[str] = []
    in_entries = False
    for line in process.stdout.splitlines():
        if line.startswith("----------"):
            in_entries = True
            continue
        if in_entries and line.startswith("Path = "):
            value = line.split("=", 1)[1].strip().replace("\\", "/")
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ProvenanceError(f"unsafe path in archive {archive.name}: {value!r}")
            paths.append(value)
    if not paths:
        raise ProvenanceError(f"7-Zip found no entries in {archive}")
    return paths


def _extract_7zip(
    archive: Path,
    destination: Path,
    *,
    include_patterns: Sequence[str] | None = None,
) -> None:
    executable = _find_7zip()
    _list_7zip_paths(archive, executable)
    destination.mkdir(parents=True, exist_ok=True)
    command = [executable, "x", str(archive), f"-o{destination}", "-y", "-bd"]
    if include_patterns:
        command.extend(f"-ir!{pattern}" for pattern in include_patterns)
    print(f"[extract] {archive.name} -> {destination}")
    subprocess.run(command, check=True)


def _copy_tree_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(child), str(target))


def _ready_marker_matches(path: Path, dataset: PrivateBundleDataset) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("bundle_files") == [
        {"name": item.name, "size": item.size, "sha256": item.sha256} for item in dataset.files
    ]


def _write_licence_marker(
    target: Path,
    dataset: PrivateBundleDataset,
    *,
    acknowledge_open_licences: bool,
    acknowledge_z24_private_terms: bool,
    acknowledge_qugs_private_rights: bool,
) -> None:
    gate = dataset.access_gate
    acknowledged = False
    note = ""
    if gate == "open_licence":
        acknowledged = acknowledge_open_licences
        note = "Open publisher licence acknowledged for private research ingestion."
    elif gate == "z24_private_terms":
        acknowledged = acknowledge_z24_private_terms
        note = (
            "User confirms prior authorised Z24 access and agrees that data remain private, non-commercial, "
            "citation-bound, and excluded from every exported artifact."
        )
    elif gate == "qugs_private_rights":
        acknowledged = acknowledge_qugs_private_rights
        note = (
            "User confirms they have a lawful private-research basis for using QUGS; this tool does not grant "
            "redistribution or reusable-data rights."
        )
    elif gate == "metadata_only":
        return
    else:
        raise DataContractError(f"unknown private bundle access gate {gate!r}")
    if not acknowledged:
        raise ProvenanceError(
            f"{dataset.dataset_id}: required access acknowledgement {gate!r} was not supplied"
        )
    atomic_write_json(
        target / ".licence_accepted.json",
        {
            "dataset_id": dataset.dataset_id,
            "access_gate": gate,
            "acknowledged_utc": utc_now_iso(),
            "private_drive_ingestion": True,
            "note": note,
        },
    )


def _require_access_acknowledgement(
    dataset: PrivateBundleDataset,
    *,
    acknowledge_open_licences: bool,
    acknowledge_z24_private_terms: bool,
    acknowledge_qugs_private_rights: bool,
) -> None:
    acknowledged = {
        "open_licence": acknowledge_open_licences,
        "z24_private_terms": acknowledge_z24_private_terms,
        "qugs_private_rights": acknowledge_qugs_private_rights,
        "metadata_only": True,
    }.get(dataset.access_gate)
    if acknowledged is None:
        raise DataContractError(f"unknown private bundle access gate {dataset.access_gate!r}")
    if not acknowledged:
        raise ProvenanceError(
            f"{dataset.dataset_id}: required access acknowledgement {dataset.access_gate!r} was not supplied"
        )


def prepare_private_drive_dataset(
    *,
    client: GoogleDriveReadOnlyClient,
    folder_files: Mapping[str, DriveFileMetadata],
    folder_id: str,
    dataset: PrivateBundleDataset,
    public_root: Path,
    archive_cache_root: Path,
    acknowledge_open_licences: bool,
    acknowledge_z24_private_terms: bool,
    acknowledge_qugs_private_rights: bool,
    overwrite: bool,
    delete_archives: bool,
    minimum_free_bytes: int,
) -> dict[str, Any]:
    target = public_root / dataset.target_subdir
    ready_marker = target / ".private_drive_ready.json"
    _require_access_acknowledgement(
        dataset,
        acknowledge_open_licences=acknowledge_open_licences,
        acknowledge_z24_private_terms=acknowledge_z24_private_terms,
        acknowledge_qugs_private_rights=acknowledge_qugs_private_rights,
    )
    if target.exists() and _ready_marker_matches(ready_marker, dataset) and not overwrite:
        print(f"[drive] {dataset.dataset_id}: already prepared and checksum-locked")
        return {
            "dataset_id": dataset.dataset_id,
            "status": "cached",
            "target": str(target),
            "training_eligible": dataset.training_eligible,
            "ineligibility_reason": dataset.ineligibility_reason,
        }

    missing = [item.name for item in dataset.files if item.name not in folder_files]
    if missing:
        raise ProvenanceError(f"Drive folder is missing {dataset.dataset_id} bundle files: {missing}")
    for expected in dataset.files:
        actual = folder_files[expected.name]
        if actual.size != expected.size:
            raise ProvenanceError(
                f"Drive file {expected.name} has size {actual.size}; expected {expected.size}"
            )

    required_peak = dataset.archive_bytes + dataset.source_bytes + minimum_free_bytes
    free = shutil.disk_usage(public_root).free
    if free < required_peak:
        raise RuntimeError(
            f"{dataset.dataset_id}: insufficient temporary disk. Conservative requirement is "
            f"{required_peak / (1024**3):.1f} GiB; available {free / (1024**3):.1f} GiB. "
            "Prepare fewer datasets per Colab session or use a larger runtime disk."
        )

    dataset_cache = archive_cache_root / dataset.dataset_id
    if overwrite and dataset_cache.exists():
        shutil.rmtree(dataset_cache)
    dataset_cache.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for expected in dataset.files:
        downloaded.append(
            client.download_resumable(
                folder_files[expected.name],
                dataset_cache / expected.name,
                expected_size=expected.size,
                expected_sha256=expected.sha256,
            )
        )

    stage = public_root / f".{dataset.dataset_id}.extracting"
    if stage.exists():
        shutil.rmtree(stage)
    _extract_7zip(
        dataset_cache / dataset.final_archive,
        stage,
        include_patterns=dataset.include_patterns,
    )
    source_prefix = stage / dataset.source_prefix
    if not source_prefix.exists():
        raise ProvenanceError(
            f"selective extraction did not create expected prefix {dataset.source_prefix!r}"
        )
    normalized = public_root / f".{dataset.dataset_id}.normalized"
    if normalized.exists():
        shutil.rmtree(normalized)
    normalized.mkdir(parents=True)
    _copy_tree_contents(source_prefix, normalized)
    shutil.rmtree(stage)

    for nested_relative in dataset.nested_archives:
        nested = normalized / nested_relative
        if not nested.exists():
            raise ProvenanceError(f"configured nested archive was not extracted: {nested_relative}")
        _extract_7zip(nested, normalized)

    if target.exists():
        if not overwrite:
            raise FileExistsError(f"target exists but does not match the locked bundle marker: {target}")
        shutil.rmtree(target)
    normalized.replace(target)

    existing_receipt = target / "download_receipt.json"
    if existing_receipt.exists():
        existing_receipt.replace(target / "publisher_download_receipt.json")
    receipt = {
        "schema_version": "vibrogemma.private-drive-receipt.v1",
        "created_utc": utc_now_iso(),
        "dataset_id": dataset.dataset_id,
        "download": {
            "source": "google_drive_private_folder",
            "folder_id_sha256": _identifier_sha256(folder_id),
            "files": [
                {
                    "name": item.name,
                    "size": item.size,
                    "sha256": item.sha256,
                    "drive_file_id_sha256": _identifier_sha256(folder_files[item.name].file_id),
                    "modified_time": folder_files[item.name].modified_time,
                }
                for item in dataset.files
            ],
        },
        "extraction": {
            "source_prefix": dataset.source_prefix,
            "include_patterns": list(dataset.include_patterns),
            "nested_archives": list(dataset.nested_archives),
            "redistribution_forbidden": dataset.redistribution_forbidden,
        },
        "training_eligible": dataset.training_eligible,
        "ineligibility_reason": dataset.ineligibility_reason,
    }
    atomic_write_json(target / "download_receipt.json", receipt)
    _write_licence_marker(
        target,
        dataset,
        acknowledge_open_licences=acknowledge_open_licences,
        acknowledge_z24_private_terms=acknowledge_z24_private_terms,
        acknowledge_qugs_private_rights=acknowledge_qugs_private_rights,
    )
    marker_payload = {
        "dataset_id": dataset.dataset_id,
        "prepared_utc": utc_now_iso(),
        "bundle_files": [
            {"name": item.name, "size": item.size, "sha256": item.sha256} for item in dataset.files
        ],
        "target": str(target),
        "training_eligible": dataset.training_eligible,
        "ineligibility_reason": dataset.ineligibility_reason,
    }
    atomic_write_json(ready_marker, marker_payload)
    if delete_archives:
        shutil.rmtree(dataset_cache)
    return {
        "dataset_id": dataset.dataset_id,
        "status": "prepared",
        "target": str(target),
        "training_eligible": dataset.training_eligible,
        "ineligibility_reason": dataset.ineligibility_reason,
        "receipt": str(target / "download_receipt.json"),
        "ready_marker": str(ready_marker),
    }


def prepare_private_drive_bundle(
    *,
    bundle: PrivateDriveBundle,
    folder_id: str,
    selected_dataset_ids: Iterable[str],
    public_root: str | Path,
    archive_cache_root: str | Path,
    acknowledge_open_licences: bool = False,
    acknowledge_z24_private_terms: bool = False,
    acknowledge_qugs_private_rights: bool = False,
    overwrite: bool = False,
    delete_archives: bool = True,
    minimum_free_gib: float = 8.0,
    client: GoogleDriveReadOnlyClient | None = None,
) -> dict[str, Any]:
    selected = list(dict.fromkeys(str(item) for item in selected_dataset_ids))
    unknown = [item for item in selected if item not in bundle.datasets]
    if unknown:
        raise DataContractError(f"private Drive bundle has no dataset definitions for {unknown}")
    client = client or GoogleDriveReadOnlyClient.from_default_credentials()
    folder_files = client.list_folder(folder_id)
    public_root = Path(public_root)
    archive_cache_root = Path(archive_cache_root)
    public_root.mkdir(parents=True, exist_ok=True)
    archive_cache_root.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset_id in selected:
        results.append(
            prepare_private_drive_dataset(
                client=client,
                folder_files=folder_files,
                folder_id=folder_id,
                dataset=bundle.datasets[dataset_id],
                public_root=public_root,
                archive_cache_root=archive_cache_root,
                acknowledge_open_licences=acknowledge_open_licences,
                acknowledge_z24_private_terms=acknowledge_z24_private_terms,
                acknowledge_qugs_private_rights=acknowledge_qugs_private_rights,
                overwrite=overwrite,
                delete_archives=delete_archives,
                minimum_free_bytes=int(float(minimum_free_gib) * 1024**3),
            )
        )
    report = {
        "schema_version": "vibrogemma.private-drive-preparation-report.v1",
        "created_utc": utc_now_iso(),
        "folder_id_sha256": _identifier_sha256(folder_id),
        "bundle_name": bundle.bundle_name,
        "selected_dataset_ids": selected,
        "folder_direct_child_count": len(folder_files),
        "datasets": results,
    }
    atomic_write_json(public_root / "private_drive_preparation_report.json", report)
    return report
