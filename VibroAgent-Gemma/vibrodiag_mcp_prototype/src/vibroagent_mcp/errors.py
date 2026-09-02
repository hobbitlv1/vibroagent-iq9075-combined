"""Error formatting helpers for API and model-response payloads."""

from __future__ import annotations

from typing import Any


def format_exception_for_response(
    exc: BaseException,
    *,
    max_depth: int = 4,
    max_children: int = 5,
    max_part_chars: int = 1000,
) -> str:
    """Return a compact exception string, unwrapping Python 3.11 exception groups."""

    parts: list[str] = []
    seen: set[int] = set()

    def add(current: BaseException, *, label: str | None = None, depth: int = 0) -> None:
        if id(current) in seen:
            return
        seen.add(id(current))
        prefix = f"{label}: " if label else ""
        parts.append(prefix + _single_exception_text(current, max_part_chars=max_part_chars))
        if depth >= max_depth:
            return

        children = getattr(current, "exceptions", None)
        if isinstance(children, tuple) and children:
            for index, child in enumerate(children[:max_children], start=1):
                if isinstance(child, BaseException):
                    add(child, label=f"sub-exception {index}", depth=depth + 1)
            if len(children) > max_children:
                parts.append(f"{len(children) - max_children} more sub-exception(s) omitted")

        cause = getattr(current, "__cause__", None)
        context = None if getattr(current, "__suppress_context__", False) else getattr(current, "__context__", None)
        if isinstance(cause, BaseException):
            add(cause, label="caused by", depth=depth + 1)
        elif isinstance(context, BaseException):
            add(context, label="during handling", depth=depth + 1)

    add(exc)
    return "; ".join(part for part in parts if part).strip()


def _single_exception_text(exc: BaseException, *, max_part_chars: int) -> str:
    message = " ".join(str(exc).split())
    text = f"{exc.__class__.__name__}: {message}" if message else exc.__class__.__name__
    return _truncate(text, max_part_chars)


def _truncate(value: Any, max_chars: int) -> str:
    text = str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
