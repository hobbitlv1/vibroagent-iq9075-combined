"""OpenAI-compatible HTTP server for the GenieX Python SDK (llama.cpp/Hexagon).

The GenieX pip distribution ships the Python SDK + ``geniex-py`` CLI but not
the ``geniex serve`` HTTP server, so this thin adapter provides the same
OpenAI surface the rest of the stack already speaks (the Genie adapter
pattern), backed by ``geniex.AutoModelForCausalLM``:

    ~/geniex-venv/bin/python -u -m vibroagent_mcp.geniex_openai_server \
        --host 127.0.0.1 --port 18181

Compared to the QAIRT/Genie adapter this backend honors per-request
``max_tokens``, ``temperature``, ``stop``, ``response_format`` with
``json_object`` (llama.cpp JSON mode) or ``json_schema`` (compiled here to a
GBNF grammar — the SDK only accepts raw GBNF), and an extra-body ``grammar``
field carrying a hand-written GBNF grammar. Context length is a runtime
setting (``GENIEX_N_CTX``), not a compiled constant. A prompt the runtime
truncates at the context limit is rejected with an explicit error instead of
returning output generated from a silently mangled prompt.

Vision (EXPERIMENTAL, CPU-only, slow): started with ``--mmproj-path`` the
model loads as a GenieXVLM and requests may carry OpenAI image content parts
(``data:`` URIs only — this is a loopback adapter and must never fetch URLs).
Images are single-turn grounding, accepted ONLY in the final message and only
when that message's role is ``user``: the SDK's chat template assumes
prior-turn media lives in the KV cache, but this server resets the KV per
request, so a historical image part can never be matched to pixel data — a
follow-up must re-attach the image to its final user message. Decoded images
become temp files (the SDK takes paths) that are deleted after every request
and purged before a watchdog re-exec, since ``finally`` does not run across
``execv``.

Three board-measured (2026-07-23) constraints shape the vision path. First, an
image request on any HTP-inclusive device_map SEGFAULTS the whole process (the
CLIP tower's ops are unsupported on HTP and crash instead of falling back), so
vision is refused at startup unless the device is in an exact crash-safe
allowlist — only ``llama_cpp:cpu`` today. Second, the CPU VLM handle is
single-shot: only the first ``generate()`` after a (re)load succeeds and the
next fails ``-201201`` even for text with ``reset()``, so the server fully
reloads the handle (~3.6s) before each request. Third, each reload LEAKS ~4.9 GB
of native memory that ``geniex_vlm_destroy()`` does not reclaim, so reloads are
hard-capped (``GENIEX_RELOAD_BUDGET``) and the runtime trips a breaker before it
can OOM the box — vision then needs a process restart. Together these make CPU
vision usable only as a slow, memory-capped experimental probe, never a
production default; text serving uses the plain LLM handle on HTP and is
unaffected by any of them.

Run it with the GenieX virtualenv's Python (stdlib + ``geniex`` only; no
project dependencies are imported at runtime).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import inspect
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

DEFAULT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_0"
# 'npu' is the QAIRT plugin alias: it rejects .gguf files outright for local
# paths, and only ever worked for hub refs via the cache manifest's plugin
# hint rebinding it to llama_cpp. This adapter exists solely for the
# GGUF-on-llama.cpp path (QAIRT bundles have their own adapter), so the
# default is the explicit proven selector.
DEFAULT_DEVICE_MAP = "llama_cpp:HTP0"
# 8192 fails on HTP0: the KV cache needs a ~1 GiB fastrpc_mmap, above the
# Hexagon buffer-mapping limit. 6144 is the measured ceiling that still loads.
DEFAULT_N_CTX = 6144
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_TIMEOUT_S = 300.0
# Wall-clock ceiling for one locked native section (template+reset+generate).
# Worst legitimate case observed: ~6k-token prefill plus a 1024-token decode at
# ~11 tok/s stays under ~150 s; a section older than this is a wedged plugin
# call that will never return and permanently bricks the single-flight lock.
DEFAULT_GENERATION_WATCHDOG_S = 180.0
# A model (re)load is ~3.6s; anything past this while holding the single-flight
# lock is a wedged from_pretrained that would brick the endpoint, so the watchdog
# re-execs on it like a stuck generation (its own phase, separate limit).
DEFAULT_RELOAD_WATCHDOG_S = 90.0
# After this many consecutive reload failures the runtime trips a breaker and
# fails requests fast instead of retrying an expensive broken load every time.
DEFAULT_RELOAD_FAILURE_LIMIT = 3
# Board-measured 2026-07-23: every VLM (re)load leaks ~4.9 GB of native memory
# that geniex_vlm_destroy() does NOT reclaim, so an unbounded reload-per-request
# path OOM-kills the box after ~10 loads. This HARD cap trips the breaker first
# so vision can never crash the machine — the server must be restarted after it,
# which is acceptable for an experimental, disabled-by-default vision path.
# Counts native CONSTRUCTION ATTEMPTS (a failed/partial from_pretrained can leak
# too), not just successes. Default 2: initial handle + 2 replacement builds
# ~= 5.9 + 2*4.9 ~= 15.7 GB RSS, real headroom on a 32 GB board (4 would reach
# ~25.6 GB — too tight). The board-proven text->image->text lifecycle fits in 2.
# Raise via GENIEX_RELOAD_BUDGET only after worst-case high-water testing.
DEFAULT_RELOAD_BUDGET = 2
# Second, board-size-independent guard: refuse a reload when MemAvailable is
# below this floor, catching larger boards/backgrounds the count budget can't.
DEFAULT_MIN_MEM_AVAILABLE_BYTES = 8 * 1024 * 1024 * 1024
# 0 = derive from context as n_ctx * 6 chars/token, generous because repetitive
# prose tokenizes at ~6.5 chars/token: this is only a cheap reject for absurd
# inputs. Token-exact enforcement is the context_length rejection after
# generate(), so an over-long prompt that passes here still fails loudly.
DEFAULT_MAX_PROMPT_CHARS = 0
# Vision limits. One image per request until multi-image ordering/cleanup is
# board-tested; the byte cap is the decoded total, the pixel cap bounds the
# native decoder's W*H*4 allocation (byte caps alone don't stop decompression
# bombs). The body cap is enforced from Content-Length before the JSON parse.
DEFAULT_MAX_IMAGES = 1
DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_BODY_BYTES = 32 * 1024 * 1024
# Conservative for experimental CPU vision: only a 448x448 image (~0.2 MP) is
# board-proven, and visual-token count scales with resolution/slicing, so a big
# image can dwarf the already ~2-minute encode. 1 MP is a bounded step above the
# proven class; raise only after measuring latency/memory/visual-token behavior.
DEFAULT_MAX_IMAGE_PIXELS = 1_048_576
# Vision tokens never appear in the prompt's character count, so the cheap
# char-budget pre-check under-reserves for image requests. Reserved tokens per
# image, converted at the same 6 chars/token factor as the budget itself.
# Applied to the pre-check only: the post-generate accounting uses the
# runtime's own token counts and stop_reason, which stay authoritative.
DEFAULT_IMAGE_TOKEN_RESERVE = 1536
ADAPTER_VERSION = "0.2"
# Exact (backend, compute_unit) tuples where an image request is known crash-safe
# on this geniex/native build. Board-measured 2026-07-23: only whole-model CPU
# survives an image (HTP/hybrid segfault). Fail CLOSED for anything not listed —
# a future GPU or fixed-HTP build must earn its place here via a board smoke, not
# a prefix match. Scoped to the observed build; not a claim about future SDKs.
_VISION_SAFE_DEVICES = frozenset({("llama_cpp", "cpu")})
# Bound the number of request bodies buffered concurrently before they contend on
# the single model lock: the threaded HTTP server would otherwise let many large
# JSON bodies accumulate in memory while queued.
DEFAULT_MAX_CONCURRENT_BODIES = 4


class _SchemaGrammarError(ValueError):
    """The JSON schema uses a construct outside the GBNF subset we compile."""


# GBNF building blocks mirroring llama.cpp's canonical JSON grammar. `space` is
# a single optional whitespace char (tighter than upstream) so a constrained
# model cannot pad output with unbounded whitespace.
_GBNF_PRIMITIVES = {
    "space": r"[ \t\n]?",
    "char": r'[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])',
    "string": r'"\"" char* "\"" space',
    "integer": r'("-"? ([0-9] | [1-9] [0-9]*)) space',
    "number": r'("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space',
    "boolean": r'("true" | "false") space',
    "null": r'"null" space',
    "value": r"object | array | string | number | boolean | null",
    "object": r'"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
    "array": r'"[" space ( value ("," space value)* )? "]" space',
}
_GBNF_PRIMITIVE_DEPS = {
    "string": ("char",),
    "value": ("object", "array", "string", "number", "boolean", "null"),
    "object": ("string", "value"),
    "array": ("value",),
}


def _use_primitive(rules: dict[str, str], name: str) -> str:
    if name not in rules:
        rules[name] = _GBNF_PRIMITIVES[name]
        for dep in _GBNF_PRIMITIVE_DEPS.get(name, ()):
            _use_primitive(rules, dep)
    return name


def _gbnf_json_literal(value: Any) -> str:
    """GBNF literal matching exactly the JSON encoding of ``value``."""
    encoded = json.dumps(value, ensure_ascii=True)
    return '"' + encoded.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _compile_object_rule(schema: dict[str, Any], name: str, rules: dict[str, str]) -> str:
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return _use_primitive(rules, "object")
    required = schema.get("required")
    required_keys = {str(key) for key in required} if isinstance(required, list) else set()
    kvs: list[tuple[bool, str]] = []
    for index, (key, sub_schema) in enumerate(properties.items()):
        value_ref = _compile_schema_rule(sub_schema, f"{name}-v{index}", rules)
        kv = f'{_gbnf_json_literal(str(key))} space ":" space {value_ref}'
        kvs.append((str(key) in required_keys, kv))
    required_kvs = [kv for is_required, kv in kvs if is_required]
    optional_kvs = [kv for is_required, kv in kvs if not is_required]

    # rest(i) = any non-empty, order-preserving subset of optional_kvs[i:].
    tail_ref: str | None = None
    for index in range(len(optional_kvs) - 1, -1, -1):
        rule_name = f"{name}-o{index}"
        if tail_ref is None:
            rules[rule_name] = optional_kvs[index]
        else:
            rules[rule_name] = f'{optional_kvs[index]} ("," space {tail_ref})? | {tail_ref}'
        tail_ref = rule_name

    body = '"{" space'
    if required_kvs:
        body += " " + ' "," space '.join(required_kvs)
        if tail_ref is not None:
            body += f' ("," space {tail_ref})?'
    elif tail_ref is not None:
        body += f" ({tail_ref})?"
    body += ' "}" space'
    rules[name] = body
    return name


def _compile_schema_rule(schema: Any, name: str, rules: dict[str, str]) -> str:
    """Compile one schema node into ``rules``; returns the rule name to reference."""
    if not isinstance(schema, dict):
        raise _SchemaGrammarError(f"schema node must be an object, got {type(schema).__name__}")
    for construct in ("anyOf", "oneOf", "allOf", "$ref", "patternProperties", "pattern", "format"):
        if construct in schema:
            raise _SchemaGrammarError(f"unsupported schema construct: {construct}")
    if "const" in schema:
        rules[name] = f'{_gbnf_json_literal(schema["const"])} space'
        return name
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        rules[name] = f'({" | ".join(_gbnf_json_literal(item) for item in enum)}) space'
        return name
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        raise _SchemaGrammarError("type unions are not supported")
    if schema_type in {"string", "integer", "number", "boolean", "null"}:
        return _use_primitive(rules, str(schema_type))
    if schema_type == "array":
        items = schema.get("items")
        if items is None:
            return _use_primitive(rules, "array")
        item_ref = _compile_schema_rule(items, f"{name}-item", rules)
        rules[name] = f'"[" space ({item_ref} ("," space {item_ref})*)? "]" space'
        return name
    if schema_type == "object" or "properties" in schema:
        return _compile_object_rule(schema, name, rules)
    if schema_type is None:
        return _use_primitive(rules, "value")
    raise _SchemaGrammarError(f"unsupported schema type: {schema_type!r}")


def _json_schema_to_gbnf(schema: Any) -> str:
    rules: dict[str, str] = {}
    _use_primitive(rules, "space")
    _compile_schema_rule(schema, "root", rules)
    return "\n".join(f"{rule} ::= {body}" for rule, body in rules.items())


def _extract_json_schema(response_format: dict[str, Any]) -> Any:
    payload = response_format.get("json_schema")
    if isinstance(payload, dict):
        if isinstance(payload.get("schema"), dict):
            return payload["schema"]
        if "properties" in payload or "type" in payload:
            return payload
    if isinstance(response_format.get("schema"), dict):
        return response_format["schema"]
    raise _SchemaGrammarError("response_format.json_schema.schema is missing")


class _PayloadTooLarge(ValueError):
    """Request exceeds a size limit — mapped to HTTP 413 instead of 400."""


class _ServiceUnavailable(RuntimeError):
    """The request is well-formed but the native runtime has exhausted a
    resource (reload budget / memory headroom) — mapped to HTTP 503, not 400."""


def _mem_available_bytes() -> int | None:
    """MemAvailable from /proc/meminfo, or None if unreadable (non-Linux)."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


# Magic-number allowlist. The claimed data-URI MIME must be here AND match the
# sniffed bytes: SVG and generic binary types are rejected, and a mislabeled
# payload never reaches the native decoder under the wrong codec.
_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def _sniff_image_mime(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _image_dimensions(mime: str, data: bytes) -> tuple[int, int] | None:
    """(width, height) parsed from the container header, None if unparsable."""
    if mime == "image/png":
        if len(data) >= 24 and data[12:16] == b"IHDR":
            return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        return None
    if mime == "image/jpeg":
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                return None
            marker = data[index + 1]
            if marker == 0xFF:  # fill byte before a marker
                index += 1
                continue
            if marker in (0x01,) or 0xD0 <= marker <= 0xD8:
                index += 2
                continue
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height = int.from_bytes(data[index + 5 : index + 7], "big")
                width = int.from_bytes(data[index + 7 : index + 9], "big")
                return width, height
            index += 2 + int.from_bytes(data[index + 2 : index + 4], "big")
        return None
    if mime == "image/webp":
        if len(data) < 30:
            return None
        chunk = data[12:16]
        if chunk == b"VP8X":
            width = int.from_bytes(data[24:27], "little") + 1
            height = int.from_bytes(data[27:30], "little") + 1
            return width, height
        if chunk == b"VP8L":
            if data[20] != 0x2F:
                return None
            bits = int.from_bytes(data[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if chunk == b"VP8 ":
            if data[23:26] != b"\x9d\x01\x2a":
                return None
            width = int.from_bytes(data[26:28], "little") & 0x3FFF
            height = int.from_bytes(data[28:30], "little") & 0x3FFF
            return width, height
        return None
    return None


def _decode_data_uri_image(url: str, *, max_bytes: int, max_pixels: int) -> tuple[bytes, str]:
    """Decode and validate one image data URI; returns (bytes, file suffix).

    Error messages must never embed the URI or its payload: data URIs carry
    the full image and would otherwise leak into client errors and logs.
    """
    if url.startswith(("http://", "https://")):
        raise ValueError(
            "image_url must be a data: URI; this loopback adapter does not fetch remote URLs"
        )
    if not url.startswith("data:"):
        raise ValueError("image_url must be a base64 data: URI (data:image/...;base64,...)")
    header, separator, payload = url.partition(",")
    if not separator:
        raise ValueError("image data: URI has no comma-separated payload")
    if not header.endswith(";base64"):
        raise ValueError("image data: URI must be base64-encoded")
    claimed_mime = header[len("data:") : -len(";base64")].split(";")[0].strip().lower()
    if claimed_mime not in _IMAGE_TYPES:
        raise ValueError(
            f"unsupported image MIME type {claimed_mime!r}; allowed: {sorted(_IMAGE_TYPES)}"
        )
    if (len(payload) * 3) // 4 > max_bytes:
        raise _PayloadTooLarge(
            f"image exceeds the decoded-size limit of {max_bytes} bytes"
        )
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"image payload is not valid base64: {exc}") from None
    if len(data) > max_bytes:
        raise _PayloadTooLarge(f"image exceeds the decoded-size limit of {max_bytes} bytes")
    sniffed = _sniff_image_mime(data)
    if sniffed != claimed_mime:
        raise ValueError(
            f"image bytes do not match the declared MIME type {claimed_mime!r} "
            f"(sniffed: {sniffed or 'unknown'})"
        )
    dimensions = _image_dimensions(claimed_mime, data)
    if dimensions is None:
        raise ValueError("image dimensions could not be read from the container header")
    width, height = dimensions
    if width <= 0 or height <= 0 or width * height > max_pixels:
        raise _PayloadTooLarge(
            f"image is {width}x{height}; the pixel limit is {max_pixels}"
        )
    return data, _IMAGE_TYPES[claimed_mime]


def _normalize_messages(
    messages: list[Any], *, vlm: bool, max_images: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """OpenAI messages -> (template-ready messages, pending image descriptors).

    Text-only messages flatten content-part arrays to a joined string (for
    both LLM and VLM loads). Image parts are accepted only in the final
    message and only when its role is ``user`` — the stateless reset-per-
    request design makes earlier-turn images unmatchable to pixel data (see
    module docstring). Each pending descriptor holds the raw ``url`` plus the
    template part dict whose ``image`` value is patched with a temp-file path
    once the image is decoded inside the generation lock.
    """
    dict_messages = [m for m in messages if isinstance(m, dict)]
    normalized: list[dict[str, Any]] = []
    pending_images: list[dict[str, Any]] = []
    last_index = len(dict_messages) - 1
    for index, message in enumerate(dict_messages):
        role = str(message.get("role") or "user")
        content = message.get("content")
        if isinstance(content, str) or content is None:
            normalized.append({"role": role, "content": content if isinstance(content, str) else ""})
            continue
        if not isinstance(content, list):
            normalized.append({"role": role, "content": ""})
            continue
        parts: list[dict[str, Any]] = []
        text_chunks: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("message content parts must be objects")
            part_type = part.get("type")
            if part_type == "text":
                text = str(part.get("text") or "")
                text_chunks.append(text)
                parts.append({"type": "text", "text": text})
                continue
            if part_type == "image_url":
                if not vlm:
                    raise ValueError(
                        "this server was started without GENIEX_MMPROJ / --mmproj-path "
                        "and cannot process images"
                    )
                if index != last_index or role != "user":
                    raise ValueError(
                        "image parts are only accepted in the final message, and only "
                        "when its role is 'user': prior-turn images cannot be grounded "
                        "by this stateless adapter — re-attach the image to the final "
                        "user message"
                    )
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if not isinstance(url, str) or not url:
                    raise ValueError("image_url part is missing its url")
                if len(pending_images) >= max_images:
                    raise _PayloadTooLarge(
                        f"at most {max_images} image(s) per request are supported"
                    )
                template_part = {"type": "image", "image": ""}
                parts.append(template_part)
                pending_images.append({"url": url, "part": template_part})
                continue
            raise ValueError(
                f"unsupported content part type {str(part_type)[:32]!r}; "
                "supported: text, image_url"
            )
        has_images = any(part["type"] == "image" for part in parts)
        if has_images:
            normalized.append({"role": role, "content": parts})
        else:
            normalized.append({"role": role, "content": "\n".join(chunk for chunk in text_chunks if chunk)})
    return normalized, pending_images


def _detect_vlm_handle(model: Any) -> bool:
    """Capability probe instead of a class-name check: only the VLM handle's
    generate() takes images=[...] (GenieXLLM's does not), and test fakes
    satisfy the same contract."""
    try:
        return "images" in inspect.signature(model.generate).parameters
    except (TypeError, ValueError):
        return False


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _GenieXRuntime:
    """Single-flight wrapper around one loaded GenieX model.

    The Hexagon NPU serializes generations, so requests queue on a timed lock:
    a caller whose budget expires while queued gets a timeout error instead of
    running as orphan work after its client gave up (same policy as the Genie
    adapter).
    """

    def __init__(
        self,
        *,
        model_ref: str,
        device_map: str,
        n_ctx: int,
        max_output_tokens: int,
        timeout_s: float,
        max_prompt_chars: int,
        mmproj_path: str | None = None,
        model_label: str | None = None,
        unsafe_allow_vision_device: str | None = None,
        expect_device: str | None = None,
        expect_model_sha256: str | None = None,
        expect_mmproj_sha256: str | None = None,
        max_images: int = DEFAULT_MAX_IMAGES,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
        image_token_reserve: int = DEFAULT_IMAGE_TOKEN_RESERVE,
        image_tmpdir: str | None = None,
    ) -> None:
        self.model_ref = model_ref
        self.device_map = device_map
        self.n_ctx = int(n_ctx)
        self.max_output_tokens = int(max_output_tokens)
        self.timeout_s = float(timeout_s)
        self.max_prompt_chars = int(max_prompt_chars)
        self.mmproj_path = mmproj_path or None
        self.model_label = model_label or None
        # Exact-value danger override: the operator must name the precise
        # backend:device they are forcing (e.g. 'llama_cpp:HTP0'), so a stale
        # boolean can never silently re-enable a crashing path.
        self.unsafe_allow_vision_device = (unsafe_allow_vision_device or "").strip().lower() or None
        self.expect_device = expect_device or None
        self.expect_model_sha256 = (expect_model_sha256 or "").lower() or None
        self.expect_mmproj_sha256 = (expect_mmproj_sha256 or "").lower() or None
        self.max_images = int(max_images)
        self.max_image_bytes = int(max_image_bytes)
        self.max_image_pixels = int(max_image_pixels)
        self.image_token_reserve = int(image_token_reserve)
        self.image_tmpdir = image_tmpdir or tempfile.gettempdir()
        self._lock = threading.Lock()
        self._model: Any = None
        self.is_vlm = False
        # The VLM handle is single-shot on this build: only the first generate()
        # after a (re)load succeeds; the next fails -201201 even for text with
        # reset() (board-measured 2026-07-23 — reset does not restore native VLM
        # state, but a full reload does, ~3.6s). So a VLM handle is reloaded
        # before each request after its first generation. The plain LLM handle
        # (production HTP text) is stable across calls and never reloads.
        self._needs_model_reload = False
        self.reload_count = 0
        self.reload_attempt_count = 0
        self.reload_failure_count = 0
        self.last_reload_ms: float | None = None
        self.last_reload_error: str | None = None
        # unavailable once the reload breaker trips; requests then fail fast.
        self.runtime_unavailable = False
        # os.stat identity of the model/mmproj at load, so a reload can confirm
        # the files were not swapped since the startup hash without re-hashing GBs.
        self._file_identities: dict[str, tuple[int, int, int]] = {}
        # Set while inside _reload_model(); the watchdog reads it as its own phase.
        self._reload_started_mono: float | None = None
        self.reload_failure_limit = DEFAULT_RELOAD_FAILURE_LIMIT
        self.reload_budget = DEFAULT_RELOAD_BUDGET
        self.min_mem_available_bytes = DEFAULT_MIN_MEM_AVAILABLE_BYTES
        # When set (vision-worker deployments), a VLM server rejects text-only
        # requests so ordinary text traffic can't burn the scarce reload budget.
        self.require_image = False
        self.loaded_backend: str | None = None
        self.loaded_device: str | None = None
        self.model_sha256: str | None = None
        self.mmproj_sha256: str | None = None
        self.geniex_version: str | None = None
        self.startup_smoke_result: dict[str, Any] | None = None
        # Set while a thread is inside the locked native section; the watchdog
        # reads it to detect a plugin call that will never return.
        self._generation_started_mono: float | None = None
        self._generation_label = ""

    def _verify_artifact_hashes(self) -> None:
        """Fail closed before the model is constructed: the full-file hash is
        the identity that separates a board-proven requant from the shipped
        artifact (or a wrong embd variant with identical filename shape)."""
        if self.expect_model_sha256:
            repo = self.model_ref.partition(":")[0]
            if not os.path.isfile(repo):
                raise RuntimeError(
                    f"GENIEX_EXPECT_MODEL_SHA256 is set but the model ref is not a local file: {repo}"
                )
            self.model_sha256 = _sha256_file(repo)
            if self.model_sha256 != self.expect_model_sha256:
                raise RuntimeError(
                    f"model sha256 mismatch: file {os.path.basename(repo)} is "
                    f"{self.model_sha256}, expected {self.expect_model_sha256}"
                )
        if self.expect_mmproj_sha256:
            if not self.mmproj_path or not os.path.isfile(self.mmproj_path):
                raise RuntimeError(
                    "GENIEX_EXPECT_MMPROJ_SHA256 is set but no mmproj file is configured"
                )
            self.mmproj_sha256 = _sha256_file(self.mmproj_path)
            if self.mmproj_sha256 != self.expect_mmproj_sha256:
                raise RuntimeError(
                    f"mmproj sha256 mismatch: file {os.path.basename(self.mmproj_path)} is "
                    f"{self.mmproj_sha256}, expected {self.expect_mmproj_sha256}"
                )

    def _construct_model(self) -> Any:
        from geniex import AutoModelForCausalLM  # lazy: only the serving venv has geniex

        repo, _, precision = self.model_ref.partition(":")
        kwargs: dict[str, Any] = {}
        if self.mmproj_path:
            kwargs["mmproj_path"] = self.mmproj_path
        return AutoModelForCausalLM.from_pretrained(
            repo,
            precision=precision or None,
            device_map=self.device_map,
            n_ctx=self.n_ctx,
            **kwargs,
        )

    def _record_file_identities(self) -> None:
        """Snapshot (inode, size, mtime_ns) of the verified model/mmproj so a
        reload can cheaply confirm they were not replaced after the startup
        hash — hashing ~3.3 GB before every reload is not viable."""
        paths = {"model": self.model_ref.partition(":")[0]}
        if self.mmproj_path:
            paths["mmproj"] = self.mmproj_path
        for key, path in paths.items():
            if os.path.isfile(path):
                st = os.stat(path)
                self._file_identities[key] = (st.st_ino, st.st_size, st.st_mtime_ns)

    def _verify_file_identities_unchanged(self) -> None:
        for key, expected in self._file_identities.items():
            path = self.model_ref.partition(":")[0] if key == "model" else self.mmproj_path
            st = os.stat(path)
            if (st.st_ino, st.st_size, st.st_mtime_ns) != expected:
                raise RuntimeError(
                    f"{key} file {os.path.basename(path)} changed since startup "
                    "(inode/size/mtime differ); refusing to reload a possibly-swapped artifact"
                )

    def _reload_model(self) -> None:
        """Rebuild the poisoned VLM handle inside the single-flight lock.

        Guards a swapped artifact (TOCTOU vs the startup hash) and revalidates
        the replacement handle's type/device; on any failure the runtime is left
        without a usable model rather than falling back to the poisoned one, and
        the reload breaker trips after repeated failures."""
        # Native-leak safeguard, checked BEFORE the leaky allocation. Count
        # CONSTRUCTION ATTEMPTS, not successes: a failed/partial from_pretrained
        # can leak too, so charging only successes would let repeated failures
        # bypass the bound. A spent budget or low memory degrades vision to a
        # clean 503 "restart required" instead of OOM-killing the box.
        if self.reload_budget > 0 and self.reload_attempt_count >= self.reload_budget:
            self.runtime_unavailable = True
            self._model = None
            raise _ServiceUnavailable(
                f"reload budget exhausted ({self.reload_attempt_count}/{self.reload_budget} "
                "construction attempts); each VLM reload leaks native memory on this build, so "
                "the server must be restarted (raise GENIEX_RELOAD_BUDGET only with RAM headroom)"
            )
        available = _mem_available_bytes()
        if available is not None and available < self.min_mem_available_bytes:
            self.runtime_unavailable = True
            self._model = None
            raise _ServiceUnavailable(
                f"insufficient memory headroom for a VLM reload: MemAvailable {available // (1024*1024)} "
                f"MiB < floor {self.min_mem_available_bytes // (1024*1024)} MiB; restart required"
            )
        # Consume the construction slot before the native call: once attempted,
        # the slot is spent regardless of outcome.
        self.reload_attempt_count += 1
        self._reload_started_mono = time.monotonic()
        try:
            self._verify_file_identities_unchanged()
            # Dispose the poisoned handle BEFORE constructing the replacement:
            # correct hygiene (free the C handle we own, don't wait on GC) and it
            # caps peak to one handle during the swap. NOTE it does NOT stop the
            # leak — board-measured, ~4.9 GB is lost per reload even with close(),
            # somewhere geniex 0.3.16 does not reclaim; the reload BUDGET breaker
            # above is what actually bounds memory. A reload failure leaves no
            # usable model — the failure breaker handles that.
            old = self._model
            self._model = None
            if old is not None:
                close = getattr(old, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            replacement = self._construct_model()
            # Revalidate WITHOUT a generation (the fresh handle has one usable
            # call): same handle class and same crash-safe device as at load.
            if _detect_vlm_handle(replacement) != self.is_vlm:
                raise RuntimeError("reloaded handle changed VLM capability")
            self._model = replacement
            self._needs_model_reload = False
            self.reload_count += 1
            self.reload_failure_count = 0
            self.last_reload_error = None
        except Exception as exc:
            # Do not keep serving the poisoned handle.
            self._model = None
            self.reload_failure_count += 1
            self.last_reload_error = f"{exc.__class__.__name__}: {exc}"
            if self.reload_failure_count >= self.reload_failure_limit:
                self.runtime_unavailable = True
            raise
        finally:
            self.last_reload_ms = round((time.monotonic() - self._reload_started_mono) * 1000.0, 1)
            self._reload_started_mono = None

    def handle_state(self) -> str:
        if self.runtime_unavailable or self._model is None:
            return "unavailable"
        if self._reload_started_mono is not None:
            return "reloading"
        if self._needs_model_reload:
            return "reload_required"
        return "fresh"

    def stuck_reload(self, limit_s: float) -> tuple[float, str] | None:
        started = self._reload_started_mono
        if started is None or limit_s <= 0:
            return None
        age = time.monotonic() - started
        if age < limit_s:
            return None
        return age, "model reload"

    def load(self) -> None:
        self._verify_artifact_hashes()
        try:
            from importlib.metadata import version

            self.geniex_version = version("geniex")
        except Exception:
            self.geniex_version = None

        started = time.monotonic()
        self._model = self._construct_model()
        self.is_vlm = _detect_vlm_handle(self._model)
        meta = getattr(self._model, "_meta", None) or {}
        self.loaded_backend = meta.get("backend")
        self.loaded_device = meta.get("device")
        self._verify_expected_device()
        self._verify_vision_device()
        self._record_file_identities()
        print(
            f"GenieX model ready in {time.monotonic() - started:.1f}s: {self.model_ref} "
            f"on {self.loaded_backend}:{self.loaded_device} n_ctx={self.n_ctx} "
            f"vlm={self.is_vlm}"
        )

    def _configured_compute_unit(self) -> str:
        """The compute unit the operator asked for, from the CONFIGURED
        device_map. This is the honest intent signal: the SDK's resolved
        device is lossy — a CPU load reports device=None, not 'cpu' — so it
        cannot be trusted to discriminate CPU from HTP after the fact."""
        device_map = (self.device_map or "auto").lower()
        return device_map.split(":", 1)[1] if ":" in device_map else device_map

    def device_resolved(self) -> str:
        # llama_cpp reports device=None for a CPU load; render it as cpu so
        # /health is readable rather than 'llama_cpp:None'.
        return f"{self.loaded_backend}:{self.loaded_device or 'cpu'}"

    def _verify_expected_device(self) -> None:
        """The SDK exposes no independently observed device (profile
        backend/device are stamped from the configured meta), so this
        validates selector RESOLUTION — it catches the npu->QAIRT class of
        misroute, not physical HTP placement; placement evidence comes from
        the offload trace smoke."""
        if not self.expect_device:
            return
        expected = self.expect_device.lower()
        backend = (self.loaded_backend or "").lower()
        # A CPU load resolves to device=None; treat that as 'cpu'.
        device = (self.loaded_device or "cpu").lower()
        if ":" in expected:
            want_backend, want_device = expected.split(":", 1)
            matches = backend == want_backend and device == want_device
        elif expected == backend:
            matches = True  # matched a backend name (e.g. 'llama_cpp')
        else:
            matches = device == expected  # a bare compute unit (e.g. 'htp0', 'cpu')
        if not matches:
            raise RuntimeError(
                f"device resolution mismatch: loaded {self.device_resolved()}, "
                f"expected {self.expect_device} (GENIEX_EXPECT_DEVICE)"
            )

    def _vision_device_key(self) -> tuple[str, str]:
        """Normalized (backend, compute_unit) the operator configured. A CPU
        load reports device=None, so this uses the CONFIGURED device_map — the
        honest intent signal — with 'llama_cpp' as the plugin default."""
        backend = (self.loaded_backend or "llama_cpp").lower()
        return backend, self._configured_compute_unit()

    def _vision_override_matches(self) -> bool:
        """True when the operator named this exact device in the danger
        override (e.g. GENIEX_UNSAFE_ALLOW_VISION_DEVICE=llama_cpp:HTP0)."""
        if not self.unsafe_allow_vision_device:
            return False
        backend, unit = self._vision_device_key()
        override = self.unsafe_allow_vision_device
        return override in {f"{backend}:{unit}", unit}

    def vision_status(self) -> str:
        """Support level for image requests, for /health and the request guard.

        disabled — no mmproj loaded. experimental_cpu — the sole board-verified
        crash-safe config. blocked_current_build — HTP/hybrid/etc., refused.
        unsafe_override — operator forced a non-allowlisted device by exact name."""
        if not self.is_vlm:
            return "disabled"
        if self._vision_device_key() in _VISION_SAFE_DEVICES:
            return "experimental_cpu"
        if self._vision_override_matches():
            return "unsafe_override"
        return "blocked_current_build"

    def _verify_vision_device(self) -> None:
        """Fail closed at startup unless vision is on a known crash-safe device.

        Board-measured 2026-07-23: an image request on any HTP-inclusive
        device_map (HTP0, hybrid) hard-SEGFAULTS the native plugin — the CLIP
        vision tower hits unsupported ops (IM2COL/UPSCALE/f32 MUL_MAT) during
        image-slice encode and crashes rather than falling back to CPU (the
        exact internal fault — allocator vs kernel vs sched — is unproven, but
        the guard does not depend on which). A segfault kills the whole daemon,
        so the generation watchdog (hang-only) cannot help. Only an exact
        allowlist of (backend, compute_unit) tuples is trusted; everything else
        fails closed, including future GPU/HTP builds until they earn a slot.
        The escape hatch is GENIEX_UNSAFE_ALLOW_VISION_DEVICE naming the precise
        device; it is never set by the production wrapper."""
        if not self.is_vlm:
            return
        status = self.vision_status()
        if status in {"experimental_cpu", "unsafe_override"}:
            return
        backend, unit = self._vision_device_key()
        raise RuntimeError(
            f"vision (mmproj) is loaded but device {backend}:{unit} (device_map="
            f"{self.device_map!r}) is not in the vision-safe allowlist {sorted(_VISION_SAFE_DEVICES)}; "
            "an image request there segfaults the plugin on this build. Serve vision with "
            "GENIEX_DEVICE_MAP=llama_cpp:cpu, or set "
            f"GENIEX_UNSAFE_ALLOW_VISION_DEVICE={backend}:{unit} to force it (unsafe)."
        )

    def startup_smoke(self) -> dict[str, Any]:
        """One real generation through the normal request path before the
        socket binds. A single requested token would not prove decode (the
        first token is sampled from prefill logits), so ask for several and
        read the runtime-reported completion count; the SDK has no
        min_new_tokens, so an immediate EOS degrades honestly to
        decode_proven=false instead of failing a healthy model."""
        response = self.chat(
            {
                "messages": [
                    {"role": "user", "content": "Reply with exactly these two words: continue monitoring."}
                ],
                "max_tokens": 8,
                "temperature": 0,
            }
        )
        metrics = response.get("geniex_metrics") or {}
        reported = int(metrics.get("completion_tokens_reported") or 0)
        result = {
            "ok": True,
            "decode_proven": reported >= 2,
            "completion_tokens_reported": reported,
            "decode_speed_tps": metrics.get("decode_speed_tps"),
            "ttft_ms": metrics.get("ttft_ms"),
        }
        if not result["decode_proven"]:
            result["note"] = (
                "runtime reported <2 completion tokens; treat as a load/prefill smoke only"
            )
        self.startup_smoke_result = result
        return result

    def health_payload(self) -> dict[str, Any]:
        """No generation here: /health must stay cheap enough to poll while a
        long request holds the single-flight lock."""
        model_file = self.model_ref.partition(":")[0]
        return {
            # A poisoned-but-reloadable handle is still 'ok' (the next request
            # reloads); only a tripped reload breaker is 'unavailable'.
            "status": "unavailable" if self.runtime_unavailable else "ok",
            "model": os.path.basename(model_file),
            "model_label": self.model_label,
            "model_sha256": self.model_sha256,
            "mmproj": os.path.basename(self.mmproj_path) if self.mmproj_path else None,
            "mmproj_sha256": self.mmproj_sha256,
            "vlm": self.is_vlm,
            # Support level, not a plain ready flag: webchat must not turn on an
            # image UI for 'blocked_current_build' or treat 'experimental_cpu' as
            # production-ready. vision_available is true only for the crash-safe
            # config; an unsafe override is reported distinctly, never as safe.
            "vision_status": self.vision_status(),
            "vision_available": self.vision_status() == "experimental_cpu",
            "vision_note": (
                "VLM handle is single-shot on this build; the server reloads it "
                f"(~3.3s) before each native generate, but each reload leaks ~4.9 GB "
                f"native memory, so capacity is the initial handle plus {self.reload_budget} "
                "replacement-handle generation attempts (text-only calls count too), "
                "then a restart is required. Slow (~2 min/image), experimental, off by default."
            )
            if self.is_vlm
            else None,
            "handle_state": self.handle_state(),
            "vlm_reuse_policy": "reload_before_each_generate" if self.is_vlm else "reuse",
            "reload_count": self.reload_count,  # successful reloads
            "construction_attempts": self.reload_attempt_count,  # authoritative budget unit
            "reload_failure_count": self.reload_failure_count,
            "reload_budget": self.reload_budget,
            "generation_attempts_remaining": (
                max(0, self.reload_budget - self.reload_attempt_count) if self.is_vlm else None
            ),
            "last_reload_ms": self.last_reload_ms,
            "last_reload_error": self.last_reload_error,
            "backend": self.loaded_backend,
            # Configured/resolved selector only: no observed-device field
            # exists at the SDK's Python layer (see load()).
            "device_configured": self.device_map,
            "device_resolved": self.device_resolved(),
            "n_ctx": self.n_ctx,
            "max_images": self.max_images if self.is_vlm else 0,
            "startup_smoke": self.startup_smoke_result,
            "adapter_version": ADAPTER_VERSION,
            "geniex_version": self.geniex_version,
            "git_commit": os.environ.get("VIBROAGENT_GIT_COMMIT") or None,
        }

    def stuck_generation(self, limit_s: float) -> tuple[float, str] | None:
        """Return (age_s, label) when the in-flight native section exceeds limit_s."""
        started = self._generation_started_mono
        if started is None or limit_s <= 0:
            return None
        age = time.monotonic() - started
        if age < limit_s:
            return None
        return age, self._generation_label

    def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        normalized_messages, pending_images = _normalize_messages(
            messages, vlm=self.is_vlm, max_images=self.max_images
        )
        if not normalized_messages:
            raise ValueError("messages must contain at least one message object")

        budget_s = _coerce_float(request.get("timeout_s"), self.timeout_s)
        budget_s = max(1.0, min(self.timeout_s, budget_s))
        max_tokens = _coerce_int(request.get("max_tokens"), self.max_output_tokens)
        max_tokens = max(1, min(self.max_output_tokens, max_tokens))
        temperature = max(0.0, _coerce_float(request.get("temperature"), 0.0))
        # The C plugin's default sampler is greedy (top-k 1), which silently turns
        # temperature and seed into no-ops: sampled calls repeat verbatim unless the
        # caller also opens the sampler (top_k/top_p) and varies the seed explicitly.
        seed = max(0, _coerce_int(request.get("seed"), 0))
        top_p = max(0.0, min(1.0, _coerce_float(request.get("top_p"), 0.0)))
        top_k = max(0, _coerce_int(request.get("top_k"), 0))
        stop = request.get("stop")
        stop_list = [str(item) for item in stop] if isinstance(stop, list) else ([str(stop)] if isinstance(stop, str) else [])

        json_mode = False
        grammar_source: str | None = None
        schema_fallback: str | None = None
        grammar = request.get("grammar")
        grammar_text = str(grammar) if isinstance(grammar, str) and grammar.strip() else None
        if grammar_text is not None:
            grammar_source = "request_grammar"
        response_format = request.get("response_format")
        if grammar_text is None and isinstance(response_format, dict):
            response_format_type = response_format.get("type")
            if response_format_type == "json_schema":
                # The SDK only takes raw GBNF, so compile the schema here. A
                # schema outside the compilable subset degrades to JSON mode,
                # flagged in geniex_metrics so the downgrade is diagnosable.
                try:
                    grammar_text = _json_schema_to_gbnf(_extract_json_schema(response_format))
                    grammar_source = "json_schema"
                except _SchemaGrammarError as exc:
                    json_mode = True
                    schema_fallback = str(exc)
            elif response_format_type == "json_object":
                json_mode = True

        queued_at = time.monotonic()
        acquired = self._lock.acquire(timeout=budget_s)
        if not acquired:
            raise TimeoutError("GenieX runtime was busy until the request timeout expired")
        queue_wait_s = time.monotonic() - queued_at
        image_paths: list[str] = []
        try:
            if self.runtime_unavailable:
                raise _ServiceUnavailable(
                    f"runtime unavailable (reloads: {self.reload_attempt_count} attempted, "
                    f"{self.reload_failure_count} failed; last error: {self.last_reload_error}); "
                    "restart required"
                )
            if self.require_image and self.is_vlm and not pending_images:
                raise ValueError(
                    "this vision-worker endpoint requires an image; text-only requests are "
                    "rejected (GENIEX_VISION_REQUIRE_IMAGE) so they do not spend the reload budget"
                )
            # The VLM handle is single-shot: reload it (own watchdog phase,
            # before the generation marker) if a prior generation poisoned it,
            # so template use and generation below both run on the fresh handle.
            # A failed reload leaves _model None and raises — never falls back to
            # the poisoned handle.
            if self._needs_model_reload:
                self._reload_model()
            if self._model is None:
                raise RuntimeError("GenieX model is not loaded")
            self._generation_label = (
                f"max_tokens={max_tokens} grammar={bool(grammar_text)} temperature={temperature} "
                f"images={len(pending_images)}"
            )
            if pending_images and self.vision_status() not in {"experimental_cpu", "unsafe_override"}:
                # Defense in depth: the startup guard already refused an unsafe
                # vision device, but re-check before the native image call so a
                # future refactor cannot route an image to a crashing device.
                raise RuntimeError(
                    "refusing an image request on a non-vision-safe device "
                    f"(vision_status={self.vision_status()})"
                )
            # Decode inside the single-flight gate: at most one request's
            # decoded image bytes are transient at a time, so image memory is
            # bounded without a second admission gate.
            for pending in pending_images:
                data, suffix = _decode_data_uri_image(
                    pending["url"],
                    max_bytes=self.max_image_bytes,
                    max_pixels=self.max_image_pixels,
                )
                descriptor, path = tempfile.mkstemp(dir=self.image_tmpdir, suffix=suffix)
                image_paths.append(path)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                pending["part"]["image"] = path
            prompt = self._model._apply_chat_template(  # noqa: SLF001 - SDK's template entry point
                normalized_messages,
                True,
                False,
                None,
            )
            # Vision tokens are invisible to the char count; shrink the cheap
            # pre-check budget so image requests keep the same headroom. The
            # post-generate token accounting stays authoritative.
            prompt_char_budget = self.max_prompt_chars - (
                len(image_paths) * self.image_token_reserve * 6
            )
            if len(prompt) > prompt_char_budget:
                raise ValueError(
                    f"prompt is {len(prompt)} characters, above the effective budget of "
                    f"{prompt_char_budget} (GENIEX_MAX_PROMPT_CHARS={self.max_prompt_chars}, "
                    f"{len(image_paths)} image(s) reserved)"
                )
            # Fresh KV per request: correctness over prefix reuse.
            self._model.reset()
            generate_kwargs: dict[str, Any] = {}
            if self.is_vlm:
                # Always explicit for a VLM handle — [] on text-only requests —
                # and never present for an LLM handle, whose generate() does
                # not take the parameter.
                generate_kwargs["images"] = image_paths
                # Calling a VLM generate() poisons the handle whether it returns
                # or raises, so mark it for reload now. Pre-generate validation
                # errors above never reach here, so they don't force a reload.
                self._needs_model_reload = True
            # Generation marker set right before the native call so the watchdog
            # times only the generate, not the reload/decode/template above.
            self._generation_started_mono = time.monotonic()
            output = self._model.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                stop=stop_list or None,
                grammar=grammar_text,
                json_mode=json_mode and grammar_text is None,
                **generate_kwargs,
            )
        finally:
            for path in image_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            self._generation_started_mono = None
            self._lock.release()

        profile = getattr(output, "profile", None)
        stop_reason = getattr(profile, "stop_reason", None)
        prompt_tokens_reported = int(getattr(profile, "prompt_tokens", 0) or 0)
        completion_tokens_reported = int(getattr(profile, "generated_tokens", 0) or 0)
        # llama.cpp context-shifts instead of failing: an over-long prompt (or a
        # generation that outgrows the window) silently evicts the oldest KV
        # entries and answers from the mangled remainder, verified on-device
        # (an 11895-token prompt "succeeds" at n_ctx=6144). The runtime's own
        # token counts are the evidence; never present such output as an answer.
        context_overflowed = stop_reason == "context_length" or (
            prompt_tokens_reported > 0
            and prompt_tokens_reported + completion_tokens_reported > self.n_ctx
        )
        if context_overflowed:
            raise ValueError(
                f"prompt + completion overflowed n_ctx={self.n_ctx} "
                f"(prompt_tokens={prompt_tokens_reported or None}, "
                f"completion_tokens={completion_tokens_reported or None}, stop_reason={stop_reason}); "
                "the runtime silently drops the oldest context when this happens, so the output "
                "is rejected; shorten the prompt or start the server with a larger GENIEX_N_CTX"
            )
        prompt_tokens = prompt_tokens_reported or max(1, len(prompt) // 4)
        completion_tokens = completion_tokens_reported or max(0, len(output.text) // 4)
        ttft_us = getattr(profile, "ttft", None)
        now = int(time.time())
        return {
            "id": f"chatcmpl-geniex-{now}",
            "object": "chat.completion",
            "created": now,
            "model": str(request.get("model") or self.model_ref),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output.text},
                    "finish_reason": "length" if stop_reason == "length" else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "geniex_metrics": {
                "backend": getattr(profile, "backend", None) or self.loaded_backend,
                "device": getattr(profile, "device", None) or self.loaded_device,
                "n_ctx": self.n_ctx,
                "queue_wait_s": round(max(0.0, queue_wait_s), 3),
                # profile.ttft is microseconds (SDK formats it with _format_us)
                "ttft_ms": round(ttft_us / 1000.0, 1) if ttft_us else None,
                "prefill_speed_tps": getattr(profile, "prefill_speed", None),
                "decode_speed_tps": getattr(profile, "decode_speed", None),
                "stop_reason": stop_reason,
                "grammar_constrained": bool(grammar_text),
                "grammar_source": grammar_source,
                "json_schema_fallback": schema_fallback,
                "json_mode": bool(json_mode and grammar_text is None),
                # Runtime-reported counts, 0 when the profile is absent — the
                # usage block above falls back to char estimates, so this is
                # the only place decode execution is distinguishable from an
                # estimate (the startup smoke relies on it).
                "prompt_tokens_reported": prompt_tokens_reported,
                "completion_tokens_reported": completion_tokens_reported,
                "images": len(image_paths),
            },
        }


class GenieXOpenAIHandler(BaseHTTPRequestHandler):
    server_version = "VibroAgentGenieXOpenAI/0.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path.rstrip("/")
        runtime: _GenieXRuntime = self.server.geniex_runtime  # type: ignore[attr-defined]
        if path in ("/health", "/v1/health"):
            # Readiness: 503 once the runtime is unavailable (breaker tripped);
            # /livez stays 200 while the process is alive.
            payload = runtime.health_payload()
            status = HTTPStatus.SERVICE_UNAVAILABLE if runtime.runtime_unavailable else HTTPStatus.OK
            self._send_json(status, payload)
            return
        if path == "/livez":
            self._send_json(HTTPStatus.OK, {"status": "alive"})
            return
        if path == "/v1/models":
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": runtime.model_ref,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local-qualcomm-geniex",
                        }
                    ],
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path.rstrip("/")
        if path != "/v1/chat/completions":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found", "type": "not_found"}})
            return
        runtime: _GenieXRuntime = self.server.geniex_runtime  # type: ignore[attr-defined]
        max_body = getattr(self.server, "geniex_max_body_bytes", DEFAULT_MAX_BODY_BYTES)
        body_gate = getattr(self.server, "geniex_body_gate", None)
        gate_held = False
        try:
            # chunked/other transfer encodings carry no Content-Length; this
            # adapter requires a declared length so the body cap is enforceable.
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Transfer-Encoding is not supported; send a Content-Length body")
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ValueError("missing Content-Length header")
            try:
                length = int(raw_length)
            except ValueError:
                raise ValueError(f"invalid Content-Length: {raw_length!r}") from None
            if length <= 0:
                raise ValueError("missing request body")
            if length > max_body:
                raise _PayloadTooLarge(
                    f"request body is {length} bytes, above the limit of {max_body}"
                )
            # Bound how many large bodies are buffered at once: without this the
            # threaded server would read every queued 32 MiB body into memory
            # while they wait on the single model lock.
            if body_gate is not None:
                if not body_gate.acquire(timeout=runtime.timeout_s):
                    raise TimeoutError("server was saturated with in-flight requests")
                gate_held = True
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("request body must be a JSON object")
            if request.get("stream"):
                raise ValueError("stream=true is not supported by this adapter")
            self._send_json(HTTPStatus.OK, runtime.chat(request))
        except Exception as exc:
            timed_out = isinstance(exc, TimeoutError)
            unavailable = isinstance(exc, _ServiceUnavailable)
            if timed_out:
                status = HTTPStatus.GATEWAY_TIMEOUT
                error_type = "timeout_error"
            elif unavailable:
                # Valid request, exhausted native resource: 503, not 400.
                status = HTTPStatus.SERVICE_UNAVAILABLE
                error_type = "service_unavailable"
            elif isinstance(exc, _PayloadTooLarge):
                status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
                error_type = "geniex_adapter_error"
            else:
                status = HTTPStatus.BAD_REQUEST
                error_type = "geniex_adapter_error"
            self._send_json(
                status,
                {"error": {"message": f"{exc.__class__.__name__}: {exc}", "type": error_type}},
            )
        finally:
            if gate_held and body_gate is not None:
                body_gate.release()

    def log_message(self, format: str, *args: Any) -> None:
        if os.environ.get("GENIEX_ADAPTER_QUIET", "0") not in {"1", "true", "TRUE"}:
            super().log_message(format, *args)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _coerce_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if number != number or number in (float("inf"), float("-inf")):
        return float(default)
    return number


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GenieX (llama.cpp/Hexagon) OpenAI-compatible server")
    parser.add_argument("--host", default=_env_str("GENIEX_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(_env_str("GENIEX_PORT", "18181")))
    parser.add_argument("--model", default=_env_str("GENIEX_MODEL", DEFAULT_MODEL))
    parser.add_argument("--device-map", default=_env_str("GENIEX_DEVICE_MAP", DEFAULT_DEVICE_MAP))
    parser.add_argument("--n-ctx", type=int, default=int(_env_str("GENIEX_N_CTX", str(DEFAULT_N_CTX))))
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=int(_env_str("GENIEX_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS))),
    )
    parser.add_argument("--timeout-s", type=float, default=float(_env_str("GENIEX_TIMEOUT_S", str(DEFAULT_TIMEOUT_S))))
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=int(_env_str("GENIEX_MAX_PROMPT_CHARS", str(DEFAULT_MAX_PROMPT_CHARS))),
    )
    parser.add_argument(
        "--generation-watchdog-s",
        type=float,
        default=float(_env_str("GENIEX_GENERATION_WATCHDOG_S", str(DEFAULT_GENERATION_WATCHDOG_S))),
        help="re-exec the server when one native generation exceeds this wall-clock limit (0 disables)",
    )
    parser.add_argument(
        "--reload-watchdog-s",
        type=float,
        default=float(_env_str("GENIEX_RELOAD_WATCHDOG_S", str(DEFAULT_RELOAD_WATCHDOG_S))),
        help="re-exec the server when one model reload exceeds this wall-clock limit (0 disables)",
    )
    parser.add_argument(
        "--reload-budget",
        type=int,
        default=int(_env_str("GENIEX_RELOAD_BUDGET", str(DEFAULT_RELOAD_BUDGET))),
        help="max VLM construction attempts before the runtime trips a breaker (native-leak OOM safeguard)",
    )
    parser.add_argument(
        "--min-mem-available-mib",
        type=int,
        default=int(_env_str("GENIEX_MIN_MEM_AVAILABLE_MIB", str(DEFAULT_MIN_MEM_AVAILABLE_BYTES // (1024 * 1024)))),
        help="refuse a VLM reload when MemAvailable is below this (board-size-independent OOM guard)",
    )
    parser.add_argument(
        "--vision-require-image",
        action="store_true",
        default=os.environ.get("GENIEX_VISION_REQUIRE_IMAGE", "0") in {"1", "true", "TRUE"},
        help="reject text-only requests on a VLM server so they don't spend the reload budget",
    )
    parser.add_argument(
        "--mmproj-path",
        default=os.environ.get("GENIEX_MMPROJ") or "",
        help="multimodal projector GGUF; loads the model as a VLM and enables image requests",
    )
    parser.add_argument("--model-label", default=os.environ.get("GENIEX_MODEL_LABEL") or "")
    parser.add_argument(
        "--unsafe-allow-vision-device",
        default=os.environ.get("GENIEX_UNSAFE_ALLOW_VISION_DEVICE") or "",
        help="DANGER: force images on the named device (e.g. llama_cpp:HTP0); "
        "segfaults on current builds. Never set in production.",
    )
    parser.add_argument(
        "--expect-device",
        default=os.environ.get("GENIEX_EXPECT_DEVICE") or "",
        help="fail startup unless the resolved device is this (e.g. HTP0 or llama_cpp:HTP0)",
    )
    parser.add_argument("--expect-model-sha256", default=os.environ.get("GENIEX_EXPECT_MODEL_SHA256") or "")
    parser.add_argument("--expect-mmproj-sha256", default=os.environ.get("GENIEX_EXPECT_MMPROJ_SHA256") or "")
    parser.add_argument("--max-images", type=int, default=int(_env_str("GENIEX_MAX_IMAGES", str(DEFAULT_MAX_IMAGES))))
    parser.add_argument(
        "--max-concurrent-bodies",
        type=int,
        default=int(_env_str("GENIEX_MAX_CONCURRENT_BODIES", str(DEFAULT_MAX_CONCURRENT_BODIES))),
    )
    parser.add_argument(
        "--max-image-bytes",
        type=int,
        default=int(_env_str("GENIEX_MAX_IMAGE_BYTES", str(DEFAULT_MAX_IMAGE_BYTES))),
    )
    parser.add_argument(
        "--max-body-bytes",
        type=int,
        default=int(_env_str("GENIEX_MAX_BODY_BYTES", str(DEFAULT_MAX_BODY_BYTES))),
    )
    parser.add_argument(
        "--image-token-reserve",
        type=int,
        default=int(_env_str("GENIEX_IMAGE_TOKEN_RESERVE", str(DEFAULT_IMAGE_TOKEN_RESERVE))),
    )
    args = parser.parse_args(argv)

    max_prompt_chars = args.max_prompt_chars
    if max_prompt_chars <= 0:
        max_prompt_chars = max(1000, args.n_ctx * 6)

    if args.mmproj_path and not os.path.isfile(args.mmproj_path):
        print(f"mmproj file not found: {args.mmproj_path}", file=sys.stderr)
        return 1

    # Per-PROCESS scratch dir for decoded images under an adapter-owned parent.
    # A shared per-port dir purged at startup could delete a still-serving older
    # process's files if a new process starts and then fails to bind; a unique
    # per-process dir avoids that overlap. The parent is swept of stale dirs at
    # startup; this process's dir is removed at exit and before a watchdog execv
    # (finally does not survive exec).
    image_parent = os.path.join(tempfile.gettempdir(), "geniex-openai-images")
    os.makedirs(image_parent, exist_ok=True)
    image_tmpdir = tempfile.mkdtemp(dir=image_parent, prefix=f"p{args.port}-")

    runtime = _GenieXRuntime(
        model_ref=args.model,
        device_map=args.device_map,
        n_ctx=args.n_ctx,
        max_output_tokens=args.max_output_tokens,
        timeout_s=args.timeout_s,
        max_prompt_chars=max_prompt_chars,
        mmproj_path=args.mmproj_path or None,
        model_label=args.model_label or None,
        unsafe_allow_vision_device=args.unsafe_allow_vision_device or None,
        expect_device=args.expect_device or None,
        expect_model_sha256=args.expect_model_sha256 or None,
        expect_mmproj_sha256=args.expect_mmproj_sha256 or None,
        max_images=args.max_images,
        max_image_bytes=args.max_image_bytes,
        image_token_reserve=args.image_token_reserve,
        image_tmpdir=image_tmpdir,
    )
    runtime.reload_budget = args.reload_budget
    runtime.min_mem_available_bytes = args.min_mem_available_mib * 1024 * 1024
    runtime.require_image = args.vision_require_image
    print(f"Loading GenieX model {args.model} (device_map={args.device_map}, n_ctx={args.n_ctx}) ...")
    runtime.load()

    # Default: run the startup smoke for the stable LLM handle, SKIP it for a
    # VLM handle. A VLM smoke would poison the freshly-loaded handle and force a
    # re-arm reload before binding — and every reload leaks ~4.9 GB here — so the
    # smoke would spend budget and memory before the first request. Skipping
    # keeps handle #1 fresh for request 1. Force with GENIEX_STARTUP_SMOKE=1.
    smoke_env = os.environ.get("GENIEX_STARTUP_SMOKE")
    if smoke_env is not None:
        run_smoke = smoke_env not in {"0", "false", "FALSE"}
    else:
        run_smoke = not runtime.is_vlm
    if run_smoke:
        try:
            smoke = runtime.startup_smoke()
        except Exception as exc:
            print(f"STARTUP SMOKE FAILED: {exc.__class__.__name__}: {exc}", file=sys.stderr)
            return 1
        print(f"Startup smoke: {json.dumps(smoke)}")
        # A forced VLM startup smoke poisons the handle; re-arm before binding so
        # the first real request gets a fresh handle (costs one leaky reload).
        if runtime.is_vlm and runtime._needs_model_reload:
            try:
                runtime._reload_model()
            except Exception as exc:
                print(f"POST-SMOKE RELOAD FAILED: {exc.__class__.__name__}: {exc}", file=sys.stderr)
                return 1
            print(f"Post-smoke re-arm: reloaded VLM handle in {runtime.last_reload_ms} ms")
    elif runtime.is_vlm:
        print("Startup smoke skipped for VLM handle (avoids a leaky re-arm reload; "
              "the first request exercises the handle). Force with GENIEX_STARTUP_SMOKE=1.")

    server = ThreadingHTTPServer((args.host, args.port), GenieXOpenAIHandler)
    server.geniex_runtime = runtime  # type: ignore[attr-defined]
    server.geniex_max_body_bytes = args.max_body_bytes  # type: ignore[attr-defined]
    server.geniex_body_gate = (  # type: ignore[attr-defined]
        threading.BoundedSemaphore(args.max_concurrent_bodies) if args.max_concurrent_bodies > 0 else None
    )

    watchdog_limit_s = float(args.generation_watchdog_s)
    reload_watchdog_limit_s = float(args.reload_watchdog_s)
    if watchdog_limit_s > 0:
        # The plugin is closed C code with no cancel API: a wedged generate()
        # never returns, holds the single-flight lock forever, and bricks chat
        # and monitoring until a human restarts the process. Recover without
        # the human: re-exec this process in place (same pid, args, env). The
        # wedged thread and any queued clients die with a connection reset and
        # the model reloads in seconds. A wedged reload (unbounded from_pretrained)
        # holding the lock is caught the same way, on its own limit.
        relaunch_argv = [
            sys.executable,
            "-u",
            "-m",
            "vibroagent_mcp.geniex_openai_server",
            "--host", str(args.host),
            "--port", str(args.port),
            "--model", str(args.model),
            "--device-map", str(args.device_map),
            "--n-ctx", str(args.n_ctx),
            "--max-output-tokens", str(args.max_output_tokens),
            "--timeout-s", str(args.timeout_s),
            "--max-prompt-chars", str(args.max_prompt_chars),
            "--generation-watchdog-s", str(args.generation_watchdog_s),
            "--mmproj-path", str(args.mmproj_path),
            "--model-label", str(args.model_label),
            "--unsafe-allow-vision-device", str(args.unsafe_allow_vision_device),
            "--expect-device", str(args.expect_device),
            "--expect-model-sha256", str(args.expect_model_sha256),
            "--expect-mmproj-sha256", str(args.expect_mmproj_sha256),
            "--max-images", str(args.max_images),
            "--max-image-bytes", str(args.max_image_bytes),
            "--max-body-bytes", str(args.max_body_bytes),
            "--image-token-reserve", str(args.image_token_reserve),
            "--max-concurrent-bodies", str(args.max_concurrent_bodies),
            "--reload-watchdog-s", str(args.reload_watchdog_s),
            "--reload-budget", str(args.reload_budget),
            "--min-mem-available-mib", str(args.min_mem_available_mib),
            *(["--vision-require-image"] if args.vision_require_image else []),
        ]

        def _generation_watchdog() -> None:
            while True:
                time.sleep(5.0)
                stuck = runtime.stuck_generation(watchdog_limit_s)
                if stuck is None:
                    stuck = runtime.stuck_reload(reload_watchdog_limit_s)
                if stuck is None:
                    continue
                age_s, label = stuck
                print(
                    f"GENIEX WATCHDOG: {label} stuck for {age_s:.0f}s; "
                    "re-exec'ing the server to recover",
                    flush=True,
                )
                # finally blocks die with the exec; reclaim this process's
                # decoded image files before re-exec (the new process gets a
                # fresh per-process dir).
                shutil.rmtree(image_tmpdir, ignore_errors=True)
                os.execv(sys.executable, relaunch_argv)

        threading.Thread(target=_generation_watchdog, name="geniex-generation-watchdog", daemon=True).start()
        print(f"Generation watchdog armed: re-exec after {watchdog_limit_s:.0f}s of one stuck generation")

    print(f"Serving GenieX OpenAI-compatible API at http://{args.host}:{args.port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping GenieX adapter")
    finally:
        server.server_close()
        shutil.rmtree(image_tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
