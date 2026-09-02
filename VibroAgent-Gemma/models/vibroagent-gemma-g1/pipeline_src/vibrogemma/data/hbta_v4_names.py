from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Iterable

# Official HBTA v4 recording roots use one undamaged condition token (UDS)
# and eight damaged-state tokens (DS1..DS8), embedded in longer names such as
# ``MVS_P2_UDS_NM_Z_01`` and ``MVS_P1_DS3_SM_Y_01``.
_CONDITION_TOKEN_RE = re.compile(
    r"(?<![A-Z0-9])(?P<state>UDS|DS[_\- ]?[1-8])(?![A-Z0-9])",
    re.IGNORECASE,
)


def canonical_hbta_condition_family(value: object) -> str:
    """Extract ``UDS`` or ``DS1``..``DS8`` from an HBTA name or label.

    The parser is token-boundary aware, so the ``DS`` fragment inside ``UDS``
    can never be misclassified as a damaged state.
    """

    text = str(value).strip()
    match = _CONDITION_TOKEN_RE.search(text.upper())
    if match is None:
        raise ValueError(f"HBTA condition token not found in {text!r}")
    token = re.sub(r"[_\- ]+", "", match.group("state").upper())
    if token == "UDS" or re.fullmatch(r"DS[1-8]", token):
        return token
    raise ValueError(f"Unsupported HBTA condition token {token!r} in {text!r}")


def partition_unnumbered_uds_roots(root_names: Iterable[str]) -> dict[str, str]:
    """Deprecated compatibility helper: all official generic UDS roots are UDS.

    Earlier experimental code incorrectly invented UDS1/UDS2.  The official
    v4 corpus contains one UDS condition represented by ten recording roots.
    This helper remains only so older imports fail safely rather than silently
    applying the obsolete split.
    """

    roots = sorted({str(item) for item in root_names})
    invalid = [root for root in roots if canonical_hbta_condition_family(root) != "UDS"]
    if invalid:
        raise ValueError(f"Non-UDS roots passed to UDS compatibility helper: {invalid[:5]}")
    return {root: "UDS" for root in roots}


def publisher_root_group_id(panel: str, publisher_root: str) -> str:
    """Return a stable split-group identifier shared by all derived windows."""

    clean = PurePosixPath(str(publisher_root)).as_posix().strip("/")
    if not clean:
        raise ValueError("publisher_root must be non-empty")
    return f"hbta_v4|{panel}|{clean}"
