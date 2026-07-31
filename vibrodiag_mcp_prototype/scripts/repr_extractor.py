"""Deterministic codec-v1 representation primitives for codes_v3.

The live worker imports the frozen-checkpoint loader, residual-VQ extractor,
token map, and provenance helpers from this module. Production extraction is
CPU-only and deterministic.
"""


from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "models/codec_v1/best.pt"
DEFAULT_SPLIT_MANIFEST = REPOSITORY_ROOT / "models/codec_v1/split_manifest.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/public_windows"
DEFAULT_TOKEN_MAP = PROJECT_ROOT / "data/codec_pretrain/r1_token_map.json"


class ReplayError(RuntimeError):
    """The stored example could not be reproduced from the seeded builder."""


class CodeRoundTripMismatch(ReplayError):
    """Regenerated numeric/symbol codes do not match the immutable example."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: str | Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - provenance compatibility, not security
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class SourceExample:
    """One selected JSONL record and its stable reference."""

    line_index: int
    key: str
    row: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class ReplayedExample:
    """Exact post-treatment inputs captured immediately before normalization."""

    source: SourceExample
    raw_windows: list[np.ndarray]
    clean_quality_windows: dict[str, np.ndarray] = field(default_factory=dict)
    injection_by_slot: dict[str, dict[str, Any]] = field(default_factory=dict)
    replay_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def slot_names(self) -> list[str]:
        return ["baseline"] + [f"target_{i}" for i in range(1, len(self.raw_windows))]

    def slot_index(self, slot: str) -> int:
        try:
            return self.slot_names.index(slot)
        except ValueError as exc:
            raise KeyError(f"unknown slot {slot!r}; available: {self.slot_names}") from exc


@dataclass
class RepresentationBundle:
    """One waveform's complete codec-v1 representation chain."""

    raw_waveform: np.ndarray
    normalized_waveform: np.ndarray
    pre_rvq_latent: np.ndarray
    codes: np.ndarray
    dequantized_latent: np.ndarray
    reconstruction: np.ndarray
    level_db: float
    straight_through_dequant_max_abs_error: float

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "raw_waveform": self.raw_waveform,
            "normalized_waveform": self.normalized_waveform,
            "pre_rvq_latent": self.pre_rvq_latent,
            "codes": self.codes,
            "dequantized_latent": self.dequantized_latent,
            "reconstruction": self.reconstruction,
        }


def _metadata_for_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("strata") if "strata" in row else row.get("meta")
    if not isinstance(metadata, dict):
        raise ReplayError("JSONL row has neither a strata nor meta object")
    return metadata


def load_source_examples(
    jsonl_path: str | Path,
    *,
    example_ids: Iterable[str] | None = None,
    line_indices: Iterable[int] | None = None,
) -> tuple[list[SourceExample], dict[str, int]]:
    """Load selected records and count single/multi rows for exact replay.

    SFT train/validation rows have no example ID and are addressed by zero-based
    ``line_index``.  Eval rows may be addressed by ``example_id`` or line index.
    """

    path = Path(jsonl_path)
    wanted_ids = set(example_ids or [])
    wanted_lines = set(int(index) for index in (line_indices or []))
    if not wanted_ids and not wanted_lines:
        raise ValueError("provide at least one example_id or zero-based line_index")

    selected: list[SourceExample] = []
    kind_counts = {"single": 0, "multi": 0}
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            row = json.loads(line)
            metadata = _metadata_for_row(row)
            kind = metadata.get("kind")
            if kind not in kind_counts:
                raise ReplayError(f"line {line_index}: invalid prompt kind {kind!r}")
            kind_counts[kind] += 1

            example_id = row.get("example_id")
            if example_id is not None:
                if example_id in seen_ids:
                    raise ReplayError(f"duplicate example_id {example_id!r}")
                seen_ids.add(str(example_id))
            if line_index not in wanted_lines and example_id not in wanted_ids:
                continue
            key = str(example_id) if example_id is not None else f"line:{line_index}"
            selected.append(SourceExample(line_index, key, row, metadata))

    found_ids = {item.key for item in selected}
    missing_ids = wanted_ids - found_ids
    found_lines = {item.line_index for item in selected}
    missing_lines = wanted_lines - found_lines
    if missing_ids or missing_lines:
        raise KeyError(
            f"missing selectors: example_ids={sorted(missing_ids)}, "
            f"line_indices={sorted(missing_lines)}"
        )
    if len(selected) != len(wanted_ids) + len(wanted_lines):
        raise ReplayError("selectors overlap or do not identify unique source rows")
    return selected, kind_counts


def _replay_lane(path: Path, selected: Sequence[SourceExample]) -> str:
    if all("strata" in item.row for item in selected):
        return "eval"
    if any("strata" in item.row for item in selected):
        raise ReplayError("cannot mix eval and SFT row schemas in one replay")
    if path.name == "train.jsonl":
        return "train"
    if path.name == "val.jsonl":
        return "val"
    raise ReplayError(
        "SFT rows require a source named train.jsonl or val.jsonl so the RNG "
        "offset is unambiguous"
    )


def _float_lists_equal(left: Any, right: Any, atol: float = 1e-12) -> bool:
    try:
        a = np.asarray(left, dtype=np.float64)
        b = np.asarray(right, dtype=np.float64)
    except (TypeError, ValueError):
        return left == right
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=0.0, atol=atol))


def _capture_matches(source: SourceExample, metadata: dict[str, Any]) -> bool:
    wanted = source.metadata
    exact_fields = (
        "dataset",
        "kind",
        "carrier_source_rows",
        "quality_flags",
        "reference_alignment",
        "reference_alignment_reason",
    )
    for field_name in exact_fields:
        if field_name in wanted and _jsonable(metadata.get(field_name)) != wanted[field_name]:
            return False
    for field_name in ("kinds", "labels"):
        if field_name in wanted and _jsonable(metadata.get(field_name)) != wanted[field_name]:
            return False
    if "level_rel_db" in wanted and not _float_lists_equal(
        metadata.get("level_rel_db"), wanted["level_rel_db"]
    ):
        return False
    return True


def regenerate_examples(
    jsonl_path: str | Path,
    *,
    example_ids: Iterable[str] | None = None,
    line_indices: Iterable[int] | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    token_map: str | Path = DEFAULT_TOKEN_MAP,
) -> dict[str, ReplayedExample]:
    """Replay the v3 builder once and return exact waveforms for selected refs.

    The builder is run with its deterministic fake codec because only waveform
    generation is needed here.  Real codec extraction and the byte-exact code
    gate happen separately in :class:`CodecV1Extractor`.
    """

    jsonl_path = Path(jsonl_path)
    selected, kind_counts = load_source_examples(
        jsonl_path, example_ids=example_ids, line_indices=line_indices
    )
    lane = _replay_lane(jsonl_path, selected)

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import build_codes_sft as builder  # noqa: PLC0415

    if not builder._have_scipy():
        raise ReplayError("SciPy is required to replay the exact v3 quota/RNG stream")

    wanted_carriers = {
        tuple(int(value) for value in item.metadata.get("carrier_source_rows", []))
        for item in selected
    }
    if () in wanted_carriers:
        raise ReplayError("selected row has no carrier_source_rows provenance")

    captures: list[dict[str, Any]] = []
    original_build_example = builder.build_example
    original_normalize = builder.normalize_window
    original_clip = builder.corrupt_clipping
    original_dropout = builder.corrupt_dropout

    def wrapped_build_example(
        rng: np.random.Generator,
        codec: Any,
        dataset: str,
        ref_w: np.ndarray,
        target_ws: list[np.ndarray],
        treatments: list[tuple[str, str]],
        alignment_meta: dict[str, Any],
    ) -> dict[str, Any]:
        carriers = tuple(int(value) for value in alignment_meta.get("carrier_source_rows", []))
        if carriers not in wanted_carriers:
            return original_build_example(
                rng, codec, dataset, ref_w, target_ws, treatments, alignment_meta
            )

        normalized_inputs: list[np.ndarray] = []
        quality_calls: list[dict[str, Any]] = []

        def capture_normalize(waveform: np.ndarray) -> tuple[np.ndarray, float]:
            normalized_inputs.append(np.array(waveform, copy=True))
            return original_normalize(waveform)

        def capture_clip(
            waveform: np.ndarray, local_rng: np.random.Generator
        ) -> tuple[np.ndarray, Any]:
            output, injection = original_clip(waveform, local_rng)
            quality_calls.append(
                {
                    "subtype": "clipping",
                    "clean": np.array(waveform, copy=True),
                    "corrupted": np.array(output, copy=True),
                    "injection": _jsonable(injection.to_json()),
                }
            )
            return output, injection

        def capture_dropout(
            waveform: np.ndarray, local_rng: np.random.Generator
        ) -> tuple[np.ndarray, Any]:
            output, injection = original_dropout(waveform, local_rng)
            quality_calls.append(
                {
                    "subtype": "flatline",
                    "clean": np.array(waveform, copy=True),
                    "corrupted": np.array(output, copy=True),
                    "injection": _jsonable(injection.to_json()),
                }
            )
            return output, injection

        builder.normalize_window = capture_normalize
        builder.corrupt_clipping = capture_clip
        builder.corrupt_dropout = capture_dropout
        try:
            example = original_build_example(
                rng, codec, dataset, ref_w, target_ws, treatments, alignment_meta
            )
        finally:
            builder.normalize_window = original_normalize
            builder.corrupt_clipping = original_clip
            builder.corrupt_dropout = original_dropout

        expected_windows = 1 + len(target_ws)
        if len(normalized_inputs) != expected_windows:
            raise ReplayError(
                f"captured {len(normalized_inputs)} normalization inputs, "
                f"expected {expected_windows}"
            )
        quality_positions = [
            index for index, (kind, _severity) in enumerate(treatments, start=1)
            if kind == "corrupted_clean"
        ]
        if len(quality_positions) != len(quality_calls):
            raise ReplayError("quality-call count does not match treatment positions")

        clean_by_slot: dict[str, np.ndarray] = {}
        injection_by_slot: dict[str, dict[str, Any]] = {}
        for position, call in zip(quality_positions, quality_calls):
            slot = f"target_{position}"
            clean_by_slot[slot] = call["clean"]
            injection_by_slot[slot] = {
                "subtype": call["subtype"],
                "details": call["injection"],
            }
            if not np.array_equal(
                normalized_inputs[position].astype(call["corrupted"].dtype, copy=False),
                call["corrupted"],
            ):
                raise ReplayError(f"captured treated waveform differs for {slot}")

        captures.append(
            {
                "metadata": _jsonable(example["meta"]),
                "raw_windows": normalized_inputs,
                "clean_quality_windows": clean_by_slot,
                "injection_by_slot": injection_by_slot,
            }
        )
        return example

    counts = {
        "singles_train": 0,
        "multis_train": 0,
        "singles_val": 0,
        "multis_val": 0,
        "eval_singles": 0,
        "eval_multis": 0,
    }
    if lane == "train":
        counts["singles_train"] = kind_counts["single"]
        counts["multis_train"] = kind_counts["multi"]
    elif lane == "val":
        counts["singles_val"] = kind_counts["single"]
        counts["multis_val"] = kind_counts["multi"]
    else:
        counts["eval_singles"] = kind_counts["single"]
        counts["eval_multis"] = kind_counts["multi"]

    replay_log = io.StringIO()
    old_argv = sys.argv
    builder.build_example = wrapped_build_example
    try:
        with tempfile.TemporaryDirectory(prefix="codec_v1_replay_") as out_dir:
            sys.argv = [
                "build_codes_sft.py",
                "--data-dir",
                str(Path(data_dir)),
                "--token-map",
                str(Path(token_map)),
                "--out",
                out_dir,
                "--singles-train",
                str(counts["singles_train"]),
                "--multis-train",
                str(counts["multis_train"]),
                "--singles-val",
                str(counts["singles_val"]),
                "--multis-val",
                str(counts["multis_val"]),
                "--eval-singles",
                str(counts["eval_singles"]),
                "--eval-multis",
                str(counts["eval_multis"]),
                "--fake-codec",
                "--legacy-v3-quality-schedule",
            ]
            with contextlib.redirect_stdout(replay_log), contextlib.redirect_stderr(replay_log):
                builder.main()
    except Exception as exc:
        tail = replay_log.getvalue()[-4000:]
        raise ReplayError(f"v3 builder replay failed: {exc}\n{tail}") from exc
    finally:
        builder.build_example = original_build_example
        builder.normalize_window = original_normalize
        builder.corrupt_clipping = original_clip
        builder.corrupt_dropout = original_dropout
        sys.argv = old_argv

    result: dict[str, ReplayedExample] = {}
    for source in selected:
        matches = [capture for capture in captures if _capture_matches(source, capture["metadata"])]
        if len(matches) != 1:
            raise ReplayError(
                f"{source.key}: expected one replay match, found {len(matches)} "
                f"among {len(captures)} candidates"
            )
        match = matches[0]
        result[source.key] = ReplayedExample(
            source=source,
            raw_windows=match["raw_windows"],
            clean_quality_windows=match["clean_quality_windows"],
            injection_by_slot=match["injection_by_slot"],
            replay_metadata=match["metadata"],
        )
    return result


class CodeSymbolMap:
    """Bidirectional frame-major/book-minor code-symbol mapping."""

    def __init__(self, path: str | Path, *, codebooks: int, codebook_size: int):
        self.path = Path(path)
        document = json.loads(self.path.read_text(encoding="utf-8"))
        self.version = document.get("version")
        self.symbols = [entry["sym"] for entry in document["entries"]]
        expected = codebooks * codebook_size
        if len(self.symbols) != expected or len(set(self.symbols)) != expected:
            raise ValueError(f"token map must contain {expected} unique symbols")
        if any(len(symbol) != 1 for symbol in self.symbols):
            raise ValueError("every codec symbol must be exactly one Unicode code point")
        self.to_global = {symbol: index for index, symbol in enumerate(self.symbols)}
        self.codebooks = codebooks
        self.codebook_size = codebook_size

    def decode(self, text: str) -> np.ndarray:
        if len(text) % self.codebooks:
            raise ValueError("serialized code count is not divisible by codebook count")
        frames = len(text) // self.codebooks
        output = np.empty((self.codebooks, frames), dtype=np.int64)
        for index, symbol in enumerate(text):
            try:
                global_code = self.to_global[symbol]
            except KeyError as exc:
                raise ValueError(f"unknown code symbol U+{ord(symbol):04X}") from exc
            expected_book = index % self.codebooks
            actual_book, local_code = divmod(global_code, self.codebook_size)
            if actual_book != expected_book:
                raise ValueError(
                    f"symbol {index}: expected book {expected_book}, got {actual_book}"
                )
            output[actual_book, index // self.codebooks] = local_code
        return output

    def encode(self, codes: np.ndarray) -> str:
        array = np.asarray(codes, dtype=np.int64)
        if array.ndim != 2 or array.shape[0] != self.codebooks:
            raise ValueError(
                f"codes must have shape ({self.codebooks}, frames), got {array.shape}"
            )
        if np.any(array < 0) or np.any(array >= self.codebook_size):
            raise ValueError("code index outside codebook")
        return "".join(
            self.symbols[book * self.codebook_size + int(array[book, frame])]
            for frame in range(array.shape[1])
            for book in range(self.codebooks)
        )


_REFERENCE_CODES_RE = re.compile(
    r'^Reference sensor "baseline" codes: (?P<codes>\S+)$', re.MULTILINE
)
_TARGET_CODES_RE = re.compile(
    r'^Target sensor "(?P<slot>target_\d+)"[^\n]* codes: (?P<codes>\S+)$',
    re.MULTILINE,
)


def prompt_code_strings(row: dict[str, Any]) -> dict[str, str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return {}
    user_text = next(
        (message.get("content", "") for message in messages if message.get("role") == "user"),
        "",
    )
    reference = _REFERENCE_CODES_RE.search(user_text)
    if reference is None:
        return {}
    result = {"baseline": reference.group("codes")}
    for match in _TARGET_CODES_RE.finditer(user_text):
        result[match.group("slot")] = match.group("codes")
    return result


class CodecV1Extractor:
    """CPU-only loader and extractor for the frozen codec-v1 checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        *,
        split_manifest: str | Path | None = DEFAULT_SPLIT_MANIFEST,
        torch_threads: int = 8,
    ):
        if torch_threads <= 0:
            raise ValueError("torch_threads must be positive")
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))
        import torch  # noqa: PLC0415
        from codec_runtime import CodecConfig, RVQCodec  # noqa: PLC0415

        self.torch = torch
        self.checkpoint_path = Path(checkpoint)
        torch.set_num_threads(torch_threads)
        torch.use_deterministic_algorithms(True)
        checkpoint_obj = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        config = CodecConfig(
            **{
                key: tuple(value) if isinstance(value, list) else value
                for key, value in checkpoint_obj["config"].items()
            }
        )
        model = RVQCodec(config)
        model.load_state_dict(checkpoint_obj["model"])
        model.to("cpu").eval()

        if split_manifest is not None:
            external_split = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
            if checkpoint_obj.get("split_manifest") != external_split:
                raise ValueError("checkpoint embedded split manifest differs from supplied manifest")
        self.model = model
        self.config = config
        self.checkpoint = checkpoint_obj
        self.torch_threads = torch_threads
        self.provenance = {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
            "checkpoint_step": checkpoint_obj.get("step"),
            "checkpoint_val": _jsonable(checkpoint_obj.get("val")),
            "split_manifest": str(split_manifest) if split_manifest is not None else None,
            "split_manifest_sha256": (
                sha256_file(split_manifest) if split_manifest is not None else None
            ),
            "torch_version": torch.__version__,
            "torch_threads": torch_threads,
            "device": "cpu",
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
            "config": _jsonable(checkpoint_obj["config"]),
        }

    def _dequantize(self, codes: Any) -> Any:
        torch = self.torch
        batch, books, frames = codes.shape
        if books != self.config.codebooks:
            raise ValueError("unexpected codebook dimension")
        dequantized = torch.zeros(
            (batch, frames, self.config.latent_dim), dtype=torch.float32, device="cpu"
        )
        for book, quantizer in enumerate(self.model.quantizers):
            dequantized = dequantized + quantizer.embed[codes[:, book, :]]
        return dequantized.permute(0, 2, 1).contiguous()

    def extract_batch(self, raw_waveforms: Sequence[np.ndarray]) -> list[RepresentationBundle]:
        """Extract a full original example batch; never split singles/multis."""

        from codec_runtime import normalize_window  # noqa: PLC0415

        if not raw_waveforms:
            raise ValueError("raw_waveforms cannot be empty")
        normalized: list[np.ndarray] = []
        levels: list[float] = []
        raw_copies: list[np.ndarray] = []
        for index, waveform in enumerate(raw_waveforms):
            raw = np.asarray(waveform)
            if raw.shape != (self.config.window_samples,):
                raise ValueError(
                    f"waveform {index}: expected {(self.config.window_samples,)}, got {raw.shape}"
                )
            if not np.isfinite(raw).all():
                raise ValueError(f"waveform {index} contains nonfinite values")
            norm, level = normalize_window(raw)
            raw_copies.append(np.array(raw, copy=True))
            normalized.append(norm)
            levels.append(level)

        torch = self.torch
        inputs = torch.from_numpy(np.stack(normalized)).unsqueeze(1).to("cpu")
        if self.model.training:
            raise RuntimeError("codec must remain in eval mode; VQEma mutates in train mode")
        with torch.inference_mode():
            prequant = self.model.encoder(inputs)
            straight_through, _commit, codes = self.model.quantize(prequant)
            explicit_dequant = self._dequantize(codes)
            parity_error = float(
                torch.max(torch.abs(straight_through - explicit_dequant)).cpu().item()
            )
            if parity_error > 2e-6:
                raise RuntimeError(
                    f"straight-through/dequantized parity error {parity_error:.3g} exceeds 2e-6"
                )
            reconstruction = self.model.decoder(explicit_dequant)

        batch = len(raw_waveforms)
        frames = self.config.window_samples // 2**self.config.n_downsample
        expected_shapes = {
            "prequant": (batch, self.config.latent_dim, frames),
            "codes": (batch, self.config.codebooks, frames),
            "dequantized": (batch, self.config.latent_dim, frames),
            "reconstruction": (batch, 1, self.config.window_samples),
        }
        observed_shapes = {
            "prequant": tuple(prequant.shape),
            "codes": tuple(codes.shape),
            "dequantized": tuple(explicit_dequant.shape),
            "reconstruction": tuple(reconstruction.shape),
        }
        if observed_shapes != expected_shapes:
            raise RuntimeError(
                f"unexpected codec stage shapes: {observed_shapes} != {expected_shapes}"
            )

        pre_np = prequant.cpu().numpy()
        codes_np = codes.cpu().numpy()
        dequant_np = explicit_dequant.cpu().numpy()
        reconstruction_np = reconstruction[:, 0, :].cpu().numpy()
        bundles: list[RepresentationBundle] = []
        for index in range(len(raw_waveforms)):
            bundle = RepresentationBundle(
                raw_waveform=raw_copies[index],
                normalized_waveform=normalized[index],
                pre_rvq_latent=pre_np[index],
                codes=codes_np[index],
                dequantized_latent=dequant_np[index],
                reconstruction=reconstruction_np[index],
                level_db=float(levels[index]),
                straight_through_dequant_max_abs_error=parity_error,
            )
            for name, array in bundle.arrays().items():
                if not np.isfinite(array).all():
                    raise RuntimeError(f"nonfinite values in extracted stage {name}")
            bundles.append(bundle)
        return bundles

    def decode_prequant_batch(self, prequant_latents: Sequence[np.ndarray]) -> np.ndarray:
        """Decoder readout of pre-RVQ latents for a diagnostic-only comparison."""

        latents = np.stack([np.asarray(value, dtype=np.float32) for value in prequant_latents])
        expected = (self.config.latent_dim, self.config.window_samples // 2**self.config.n_downsample)
        if latents.shape[1:] != expected:
            raise ValueError(f"prequant latents must have trailing shape {expected}")
        with self.torch.inference_mode():
            decoded = self.model.decoder(self.torch.from_numpy(latents).to("cpu"))
        output = decoded[:, 0, :].cpu().numpy()
        if not np.isfinite(output).all():
            raise RuntimeError("nonfinite prequant decoder output")
        return output


def verify_prompt_round_trip(
    row: dict[str, Any],
    bundles: Sequence[RepresentationBundle],
    symbol_map: CodeSymbolMap,
) -> dict[str, Any]:
    """Require regenerated code IDs and UTF-8 symbol bytes to match the prompt."""

    expected_strings = prompt_code_strings(row)
    if not expected_strings:
        raise CodeRoundTripMismatch("example prompt contains no anchored codec strings")
    expected_slots = ["baseline"] + [f"target_{index}" for index in range(1, len(bundles))]
    if set(expected_strings) != set(expected_slots):
        raise CodeRoundTripMismatch(
            f"prompt slots {sorted(expected_strings)} differ from replay slots {expected_slots}"
        )

    checks: dict[str, Any] = {}
    for slot, bundle in zip(expected_slots, bundles):
        expected_text = expected_strings[slot]
        expected_codes = symbol_map.decode(expected_text)
        regenerated_text = symbol_map.encode(bundle.codes)
        shapes_equal = bundle.codes.shape == expected_codes.shape
        ids_equal = shapes_equal and bool(np.array_equal(bundle.codes, expected_codes))
        bytes_equal = regenerated_text.encode("utf-8") == expected_text.encode("utf-8")
        checks[slot] = {
            "shape_equal": shapes_equal,
            "numeric_codes_equal": ids_equal,
            "utf8_symbol_bytes_equal": bytes_equal,
            "tokens": len(expected_text),
            "expected_utf8_sha256": hashlib.sha256(expected_text.encode("utf-8")).hexdigest(),
            "regenerated_utf8_sha256": hashlib.sha256(
                regenerated_text.encode("utf-8")
            ).hexdigest(),
            "codes_i32le_sha256": hashlib.sha256(
                np.asarray(bundle.codes, dtype="<i4").tobytes(order="C")
            ).hexdigest(),
        }
        if not ids_equal or not bytes_equal:
            mismatch = (
                np.argwhere(bundle.codes != expected_codes)
                if shapes_equal else np.empty((0, 2), dtype=np.int64)
            )
            first = mismatch[0].tolist() if mismatch.size else None
            raise CodeRoundTripMismatch(
                f"{slot}: regenerated exam codes differ; shapes "
                f"{bundle.codes.shape}/{expected_codes.shape}, first numeric mismatch={first}"
            )
    return {
        "all_slots_exact": True,
        "slots_verified": len(checks),
        "symbols_verified": sum(check["tokens"] for check in checks.values()),
        "slots": checks,
    }


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.astype(np.float64).mean()),
    }


def _write_npz(
    output: Path,
    bundle: RepresentationBundle,
    *,
    metadata: dict[str, Any],
    clean_bundle: RepresentationBundle | None,
) -> None:
    payload = dict(bundle.arrays())
    payload["level_db"] = np.asarray(bundle.level_db, dtype=np.float64)
    payload["metadata_json"] = np.asarray(
        json.dumps(_jsonable(metadata), sort_keys=True, separators=(",", ":"))
    )
    if clean_bundle is not None:
        for name, array in clean_bundle.arrays().items():
            payload[f"clean_control_{name}"] = array
        payload["clean_control_level_db"] = np.asarray(clean_bundle.level_db, dtype=np.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--example-jsonl", required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--example-id")
    selector.add_argument("--line-index", type=int, help="zero-based JSONL row")
    parser.add_argument("--slot", default="target_1")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT_MANIFEST))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--token-map", default=str(DEFAULT_TOKEN_MAP))
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--output", help="optional output .npz containing all stage arrays")
    args = parser.parse_args()

    replayed = regenerate_examples(
        args.example_jsonl,
        example_ids=[args.example_id] if args.example_id else None,
        line_indices=[args.line_index] if args.line_index is not None else None,
        data_dir=args.data_dir,
        token_map=args.token_map,
    )
    # Eval rows retain their stable example_id even when selected by line; SFT
    # rows have no ID and therefore keep the ``line:N`` key.
    key = args.example_id if args.example_id else next(iter(replayed))
    example = replayed[key]
    extractor = CodecV1Extractor(
        args.checkpoint,
        split_manifest=args.split_manifest,
        torch_threads=args.torch_threads,
    )
    bundles = extractor.extract_batch(example.raw_windows)
    symbol_map = CodeSymbolMap(
        args.token_map,
        codebooks=extractor.config.codebooks,
        codebook_size=extractor.config.codebook_size,
    )
    round_trip = verify_prompt_round_trip(example.source.row, bundles, symbol_map)
    slot_index = example.slot_index(args.slot)
    bundle = bundles[slot_index]

    clean_bundle: RepresentationBundle | None = None
    if args.slot in example.clean_quality_windows:
        clean_batch = list(example.raw_windows)
        clean_batch[slot_index] = example.clean_quality_windows[args.slot]
        clean_bundle = extractor.extract_batch(clean_batch)[slot_index]

    summary = {
        "source": str(Path(args.example_jsonl).resolve()),
        "source_line_index": example.source.line_index,
        "example_key": key,
        "slot": args.slot,
        "provenance": extractor.provenance,
        "token_map": {
            "path": str(Path(args.token_map).resolve()),
            "sha256": sha256_file(args.token_map),
            "version": symbol_map.version,
        },
        "round_trip": round_trip,
        "representations": {
            name: _array_summary(array) for name, array in bundle.arrays().items()
        },
        "level_db": bundle.level_db,
        "straight_through_dequant_max_abs_error": (
            bundle.straight_through_dequant_max_abs_error
        ),
        "matched_clean_control_included": clean_bundle is not None,
        "output": str(Path(args.output).resolve()) if args.output else None,
    }
    if args.output:
        _write_npz(
            Path(args.output),
            bundle,
            metadata=summary,
            clean_bundle=clean_bundle,
        )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
