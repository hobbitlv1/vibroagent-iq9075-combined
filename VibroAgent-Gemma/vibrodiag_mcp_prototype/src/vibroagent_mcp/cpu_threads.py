"""How many threads a compute stage of the Gemma deployment may start."""

from __future__ import annotations

import os


def thread_count(variable: str) -> int:
    """The positive integer in the named environment variable, else the cores the caller may use.

    The launcher pins each service, and the web process each of its threads, to a CPU set. The
    numeric runtimes used here size their pools from the machine's core count instead. Counting
    the calling thread's allowed cores keeps a stage inside its own set.
    """
    raw = os.environ.get(variable, "").strip()
    if raw:
        try:
            requested = int(raw)
        except ValueError:
            requested = 0
        if requested < 1:
            raise RuntimeError(f"{variable} must be a positive integer, got {raw!r}")
        return requested
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)
