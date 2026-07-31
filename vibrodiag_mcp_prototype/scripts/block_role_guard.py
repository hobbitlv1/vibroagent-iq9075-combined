"""Fail-closed, stage-aware consumer for the authoritative role ledgers.

The ledger builder owns role assignment and source-overlap auditing.  This
module is deliberately smaller: training builders use it to verify the ledger
and source-pool hashes, construct their exact block-derived row sets, and then
re-check those row sets immediately before a draw can occur.

Codec-v2 and frozen-B2 consumers remain bound to immutable ledger v2.  SFT,
real-evaluation, and challenge consumers are bound to its narrowly amended
v2.1 successor.  Callers choose only a named consumer stage; there is no API
for supplying or bypassing the corresponding authoritative hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


# These pins intentionally live outside both ledgers.  Checking only hashes
# stored inside a ledger would allow a self-consistent replacement ledger and
# matching pools to authorize themselves.
AUTHORITATIVE_LEDGER_V2_SHA256 = (
    "882b9c8f68a40b8e2f3eef612e05e3dbe235fc7fe7098ea3809c2fb85909736c"
)
AUTHORITATIVE_LEDGER_V2_1_SHA256 = (
    "244ee7d7ffda93e569d2edfe72ecdec6c91de3f82e2a064e111bb1235f75940b"
)
# Compatibility alias retained for frozen codec-v2/B2 modules.  New SFT code
# must use the named lineage stage, never reinterpret this alias as "latest".
AUTHORITATIVE_LEDGER_SHA256 = AUTHORITATIVE_LEDGER_V2_SHA256

LEDGER_LINEAGE_BY_STAGE = {
    "codec": {
        "ledger_name": "block_role_ledger_v2",
        "ledger_version": "2",
        "sha256": AUTHORITATIVE_LEDGER_V2_SHA256,
    },
    "b2": {
        "ledger_name": "block_role_ledger_v2",
        "ledger_version": "2",
        "sha256": AUTHORITATIVE_LEDGER_V2_SHA256,
    },
    "sft": {
        "ledger_name": "block_role_ledger_v2_1",
        "ledger_version": "2.1",
        "sha256": AUTHORITATIVE_LEDGER_V2_1_SHA256,
    },
    "real_eval": {
        "ledger_name": "block_role_ledger_v2_1",
        "ledger_version": "2.1",
        "sha256": AUTHORITATIVE_LEDGER_V2_1_SHA256,
    },
    "challenge": {
        "ledger_name": "block_role_ledger_v2_1",
        "ledger_version": "2.1",
        "sha256": AUTHORITATIVE_LEDGER_V2_1_SHA256,
    },
}
AUTHORITATIVE_NPZ_SHA256 = {
    "fdsn": "f2522cdd97fd19df3890bf8c8732877be3b5ba2b54d4b0a9c378264b30f8e5a4",
    "lumo": "8c0ac46904e342fbf48f68f06df1841da94fa709ffa05d4ae90bc54f5c0f21e5",
    "rt345": "44c60d73cf761844cf01aa6fb9fe4a2418f48f3065666e32912eb6a58f2578c7",
}
AUTHORITATIVE_DATASETS = tuple(AUTHORITATIVE_NPZ_SHA256)


REQUIRED_ROLES = (
    "codec_train",
    "codec_val",
    "sft_train",
    "sft_val",
    "reference_bank",
    "probe_calibration",
    "challenge_target",
    "challenge_reference",
)
ISOLATED_ROLES = (
    "probe_calibration",
    "challenge_target",
    "challenge_reference",
)
STAGE_ROLES = {
    "codec": ("codec_train", "codec_val"),
    "sft": ("sft_train", "sft_val"),
}
# A lineage name is not a hash-selection escape hatch.  Only these consumers
# have a concrete reason to inspect the listed train/validation projection.
ALLOWED_SPLIT_LINEAGE_PAIRS = {
    ("codec", "codec"),
    ("sft", "sft"),
    ("sft", "b2"),
    ("codec", "real_eval"),
    ("sft", "real_eval"),
}


def require(condition: bool, message: str) -> None:
    """Terminate rather than permit an ambiguous or contaminated draw."""
    if not condition:
        raise SystemExit(f"fail-closed: {message}")


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            data = handle.read(chunk)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def _portable_repo_path(value: Any, label: str) -> PurePosixPath:
    require(isinstance(value, str) and value != "", f"{label}: missing path")
    require("\\" not in value, f"{label}: path must use portable '/' separators")
    path = PurePosixPath(value)
    require(not path.is_absolute(), f"{label}: absolute paths are forbidden: {value}")
    require(".." not in path.parts, f"{label}: path escapes repo root: {value}")
    require(path.as_posix() == value, f"{label}: path is not canonical: {value}")
    return path


def _lineage_contract(stage: str) -> dict[str, str]:
    require(
        stage in LEDGER_LINEAGE_BY_STAGE,
        "unknown ledger lineage stage: "
        f"{stage!r}; expected one of {','.join(LEDGER_LINEAGE_BY_STAGE)}",
    )
    return LEDGER_LINEAGE_BY_STAGE[stage]


def _load_ledger(
    path: str | Path,
    lineage_stage: str,
) -> tuple[dict[str, Any], str, dict[str, str]]:
    contract = _lineage_contract(lineage_stage)
    ledger_path = Path(path)
    require(ledger_path.is_file(), f"role ledger not found: {ledger_path}")
    ledger_sha = sha256_file(ledger_path)
    require(
        ledger_sha == contract["sha256"],
        f"{lineage_stage}: role ledger sha256 mismatch: "
        f"{ledger_sha} != {contract['sha256']}",
    )
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"fail-closed: cannot read role ledger {ledger_path}: {exc}") from exc
    require(isinstance(ledger, dict), "role ledger root must be an object")
    require(ledger.get("schema_version") == 2,
            f"role ledger schema must be 2, got {ledger.get('schema_version')!r}")
    require(
        ledger.get("ledger_name") == contract["ledger_name"],
        f"{lineage_stage}: ledger_name must be {contract['ledger_name']!r}",
    )
    if contract["ledger_version"] == "2.1":
        require(ledger.get("ledger_version") == "2.1",
                "v2.1 successor must declare ledger_version='2.1'")
        require(
            ledger.get("parent_ledger_sha256") == AUTHORITATIVE_LEDGER_V2_SHA256,
            "v2.1 parent_ledger_sha256 does not bind authoritative v2",
        )
        require(
            ledger.get("parent_ledger_path") == "data/audits/block_role_ledger_v2.json",
            "v2.1 parent_ledger_path changed",
        )
        require(
            ledger.get("adopted_resolution_marker")
            == "SFTBLOCKERS-MARKER-20260714-0018",
            "v2.1 adopted resolution marker changed",
        )
        require(
            ledger.get("delivery_marker") == "LEDGER21-MARKER-20260714-0019",
            "v2.1 delivery marker changed",
        )
    else:
        require("ledger_version" not in ledger,
                "immutable ledger v2 unexpectedly declares a successor version")
        require("parent_ledger_sha256" not in ledger,
                "immutable ledger v2 unexpectedly declares a parent")
    require(ledger.get("path_base") == "repo_root",
            "role ledger path_base must be 'repo_root'")
    require(isinstance(ledger.get("npz"), dict), "role ledger missing npz map")
    require(isinstance(ledger.get("datasets"), dict), "role ledger missing datasets map")
    return ledger, ledger_sha, contract


def verify_authoritative_inputs(
    ledger_path: str | Path,
    data_dir: str | Path,
    datasets: tuple[str, ...] = AUTHORITATIVE_DATASETS,
    *,
    lineage_stage: str = "codec",
) -> dict[str, Any]:
    """Hash the ledger and all source pools before any NPZ is parsed.

    This is the first operation in codec-v2 training and gate evaluation.  The
    complete three-dataset set is mandatory; a dataset subset is not a valid
    codec-v2 run.  Returned values are suitable for run provenance.
    """
    require(
        tuple(datasets) == AUTHORITATIVE_DATASETS,
        "codec-v2 datasets must be exactly "
        f"{','.join(AUTHORITATIVE_DATASETS)} in that order",
    )
    ledger = Path(ledger_path)
    _, ledger_sha, contract = _load_ledger(ledger, lineage_stage)
    root = Path(data_dir)
    pools: dict[str, dict[str, Any]] = {}
    for dataset in AUTHORITATIVE_DATASETS:
        path = root / f"{dataset}.npz"
        require(path.is_file(), f"source pool not found: {path}")
        actual = sha256_file(path)
        expected = AUTHORITATIVE_NPZ_SHA256[dataset]
        require(
            actual == expected,
            f"{dataset}: authoritative source-pool sha256 mismatch: "
            f"{actual} != {expected}",
        )
        pools[dataset] = {
            "path": path.as_posix(),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    return {
        "ledger": {
            "path": ledger.as_posix(),
            "sha256": ledger_sha,
            "size_bytes": ledger.stat().st_size,
            "ledger_name": contract["ledger_name"],
            "ledger_version": contract["ledger_version"],
            "lineage_stage": lineage_stage,
        },
        "npz": pools,
    }


def _validate_dataset(
    *,
    dataset: str,
    pool: Any,
    data_dir: Path,
    ledger: dict[str, Any],
) -> dict[str, list[str]]:
    require(dataset in ledger["npz"], f"ledger has no npz entry for {dataset}")
    require(dataset in ledger["datasets"], f"ledger has no dataset entry for {dataset}")
    source = ledger["npz"][dataset]
    require(isinstance(source, dict), f"{dataset}: npz entry must be an object")
    relpath = _portable_repo_path(source.get("path"), f"{dataset}.npz")
    require(relpath == PurePosixPath("data/public_windows") / f"{dataset}.npz",
            f"{dataset}: unexpected source-pool path {relpath}")
    actual_path = data_dir / f"{dataset}.npz"
    require(actual_path.is_file(), f"{dataset}: source pool not found: {actual_path}")
    expected_hash = source.get("sha256")
    require(isinstance(expected_hash, str) and len(expected_hash) == 64,
            f"{dataset}: invalid source-pool sha256")
    actual_hash = sha256_file(actual_path)
    require(expected_hash == AUTHORITATIVE_NPZ_SHA256[dataset],
            f"{dataset}: ledger source-pool sha256 is not the authoritative pin")
    require(actual_hash == expected_hash,
            f"{dataset}: source-pool sha256 mismatch: {actual_hash} != {expected_hash}")

    entry = ledger["datasets"][dataset]
    require(isinstance(entry, dict), f"{dataset}: ledger entry must be an object")
    roles_obj = entry.get("roles")
    require(isinstance(roles_obj, dict), f"{dataset}: roles must be an object")
    require(set(roles_obj) == set(REQUIRED_ROLES),
            f"{dataset}: roles must be exactly {list(REQUIRED_ROLES)}")

    blocks_arr = np.asarray(pool.block).astype(str)
    actual_blocks = sorted(set(blocks_arr.tolist()))
    roles: dict[str, list[str]] = {}
    for role in REQUIRED_ROLES:
        blocks = roles_obj[role]
        require(isinstance(blocks, list) and all(isinstance(block, str) for block in blocks),
                f"{dataset}/{role}: role membership must be a string list")
        require(blocks == sorted(blocks), f"{dataset}/{role}: blocks must be sorted")
        require(len(blocks) == len(set(blocks)), f"{dataset}/{role}: duplicate block")
        unknown = sorted(set(blocks) - set(actual_blocks))
        require(not unknown, f"{dataset}/{role}: unknown blocks {unknown}")
        roles[role] = blocks
    for role in (*ISOLATED_ROLES, "reference_bank"):
        require(len(roles[role]) == 1,
                f"{dataset}/{role}: ledger v2 requires exactly one block")

    reverse: dict[str, list[str]] = {block: [] for block in actual_blocks}
    for role, blocks in roles.items():
        for block in blocks:
            reverse[block].append(role)
    reverse = {block: sorted(memberships) for block, memberships in reverse.items()}
    declared_reverse = entry.get("block_roles")
    require(declared_reverse == reverse,
            f"{dataset}: block_roles does not exactly cover/reverse the role lists")

    for train_role, val_role in (("codec_train", "codec_val"),
                                 ("sft_train", "sft_val")):
        overlap = sorted(set(roles[train_role]) & set(roles[val_role]))
        require(not overlap, f"{dataset}: {train_role}/{val_role} overlap: {overlap}")
        require(roles[train_role], f"{dataset}: {train_role} is empty")
        require(roles[val_role], f"{dataset}: {val_role} is empty")

    isolated_sets = {role: set(roles[role]) for role in ISOLATED_ROLES}
    for i, role_a in enumerate(ISOLATED_ROLES):
        for role_b in ISOLATED_ROLES[i + 1:]:
            overlap = sorted(isolated_sets[role_a] & isolated_sets[role_b])
            require(not overlap, f"{dataset}: {role_a}/{role_b} overlap: {overlap}")
    fit_or_bank = set().union(*(set(roles[role]) for role in (
        "codec_train", "codec_val", "sft_train", "sft_val", "reference_bank"
    )))
    for role in ISOLATED_ROLES:
        overlap = sorted(isolated_sets[role] & fit_or_bank)
        require(not overlap, f"{dataset}: isolated role {role} enters fit/val/bank: {overlap}")
    require(set(roles["reference_bank"]) <= set(roles["sft_train"]),
            f"{dataset}: reference_bank must be a subset of sft_train")

    require(set(reverse) == set(actual_blocks),
            f"{dataset}: ledger block universe differs from source pool")
    require(all(reverse[block] for block in actual_blocks),
            f"{dataset}: at least one source block has no role")
    require(source.get("n_windows") == int(blocks_arr.size),
            f"{dataset}: npz n_windows does not match source pool")
    require(source.get("n_blocks") == len(actual_blocks),
            f"{dataset}: npz n_blocks does not match source pool")

    counts = entry.get("role_counts")
    require(isinstance(counts, dict), f"{dataset}: missing role_counts")
    for role, blocks in roles.items():
        expected = {
            "blocks": len(blocks),
            "windows": int(np.isin(blocks_arr, blocks).sum()),
        }
        require(counts.get(role) == expected,
                f"{dataset}/{role}: role_counts mismatch; expected {expected}")
    return roles


def verify_stage_indices(
    pools: list[Any],
    guard: dict[str, Any],
    train_idx: list[tuple[int, int]],
    val_idx: list[tuple[int, int]],
) -> None:
    """Re-check concrete rows against the exact ledger roles.

    Builders call this once on construction and again immediately before their
    sampler/generator.  The second call is what protects against an intervening
    refactor accidentally appending an unapproved row.
    """
    stage = guard["stage"]
    train_role, val_role = STAGE_ROLES[stage]
    datasets = guard["datasets"]
    pool_by_name = {pool.name: i for i, pool in enumerate(pools)}

    def check_lane(lane: str, role: str, indices: list[tuple[int, int]]) -> None:
        require(len(indices) == len(set(indices)), f"{stage}_{lane}: duplicate row index")
        actual_by_dataset: dict[str, set[int]] = {dataset: set() for dataset in datasets}
        for item in indices:
            require(isinstance(item, tuple) and len(item) == 2,
                    f"{stage}_{lane}: malformed row index {item!r}")
            pool_i, row = item
            require(isinstance(pool_i, (int, np.integer)) and
                    isinstance(row, (int, np.integer)),
                    f"{stage}_{lane}: non-integer row index {item!r}")
            require(0 <= int(pool_i) < len(pools),
                    f"{stage}_{lane}: pool index out of range: {pool_i}")
            pool = pools[int(pool_i)]
            require(pool.name in actual_by_dataset,
                    f"{stage}_{lane}: unrequested dataset {pool.name}")
            require(0 <= int(row) < len(pool.block),
                    f"{stage}_{lane}: {pool.name} row out of range: {row}")
            block = str(pool.block[int(row)])
            memberships = guard["roles"][pool.name]
            for forbidden_role in ISOLATED_ROLES:
                if block in memberships[forbidden_role]:
                    require(False, f"{stage}_{lane} contains forbidden {pool.name} block "
                                   f"{block!r} (role={forbidden_role})")
            require(block in memberships[role],
                    f"{stage}_{lane} contains unassigned {pool.name} block {block!r}; "
                    f"expected role={role}")
            actual_by_dataset[pool.name].add(int(row))

        for dataset in datasets:
            pool_i = pool_by_name[dataset]
            pool = pools[pool_i]
            allowed = set(guard["roles"][dataset][role])
            expected_rows = {int(row) for row in np.flatnonzero(
                np.isin(np.asarray(pool.block).astype(str), sorted(allowed))
            )}
            actual_rows = actual_by_dataset[dataset]
            missing = len(expected_rows - actual_rows)
            extra = len(actual_rows - expected_rows)
            require(not missing and not extra,
                    f"{stage}_{lane}/{dataset}: row coverage differs from ledger "
                    f"(missing={missing}, extra={extra})")
            selected_blocks = sorted({str(pool.block[row]) for row in actual_rows})
            require(selected_blocks == sorted(allowed),
                    f"{stage}_{lane}/{dataset}: block set differs from role={role}")

    check_lane("train", train_role, train_idx)
    check_lane("val", val_role, val_idx)
    require(not (set(train_idx) & set(val_idx)), f"{stage}: train/val rows overlap")


def guarded_stage_split(
    pools: list[Any],
    data_dir: str | Path,
    datasets: tuple[str, ...],
    ledger_path: str | Path,
    stage: str,
    *,
    lineage_stage: str | None = None,
) -> dict[str, Any]:
    """Load and enforce one split projection under a named ledger lineage.

    Ordinarily the split stage is also the consumer lineage (``codec`` or
    ``sft``).  Consumers that inspect a different projection must say why:
    frozen B2 uses ``lineage_stage='b2'`` while real-evaluation uses
    ``lineage_stage='real_eval'`` for both of its codec and SFT projections.
    """
    require(stage in STAGE_ROLES, f"unknown guarded stage: {stage}")
    effective_lineage_stage = lineage_stage or stage
    _lineage_contract(effective_lineage_stage)
    require(
        (stage, effective_lineage_stage) in ALLOWED_SPLIT_LINEAGE_PAIRS,
        "forbidden split/lineage pairing: "
        f"stage={stage!r} lineage_stage={effective_lineage_stage!r}",
    )
    require(datasets and len(datasets) == len(set(datasets)),
            "datasets must be a non-empty unique sequence")
    require(len(pools) == len(datasets), "pool count differs from requested datasets")
    require(tuple(pool.name for pool in pools) == datasets,
            "pool order differs from requested datasets")

    ledger, ledger_sha, lineage_contract = _load_ledger(
        ledger_path, effective_lineage_stage
    )
    split_source = (ledger.get("assignment_policy") or {}).get("legacy_split_source")
    require(isinstance(split_source, dict), "ledger missing assignment_policy.legacy_split_source")
    require(_portable_repo_path(split_source.get("path"), "legacy_split_source") ==
            PurePosixPath("data/codec_pretrain/split_manifest.json"),
            "ledger legacy split source path changed")
    require(split_source.get("seed") == 20260708,
            "ledger legacy split seed must remain 20260708")
    require(split_source.get("val_fraction") == 0.15,
            "ledger legacy split val_fraction must remain 0.15")
    roles: dict[str, dict[str, list[str]]] = {}
    data_dir_path = Path(data_dir)
    for pool in pools:
        roles[pool.name] = _validate_dataset(
            dataset=pool.name, pool=pool, data_dir=data_dir_path, ledger=ledger
        )

    train_role, val_role = STAGE_ROLES[stage]
    train_idx: list[tuple[int, int]] = []
    val_idx: list[tuple[int, int]] = []
    manifest_datasets: dict[str, Any] = {}
    for pool_i, pool in enumerate(pools):
        blocks = np.asarray(pool.block).astype(str)
        train_blocks = roles[pool.name][train_role]
        val_blocks = roles[pool.name][val_role]
        train_rows = np.flatnonzero(np.isin(blocks, train_blocks))
        val_rows = np.flatnonzero(np.isin(blocks, val_blocks))
        train_idx.extend((pool_i, int(row)) for row in train_rows)
        val_idx.extend((pool_i, int(row)) for row in val_rows)
        manifest_datasets[pool.name] = {
            "n_windows": int(blocks.size),
            "n_blocks": len(set(blocks.tolist())),
            "train_blocks": train_blocks,
            "val_blocks": val_blocks,
            "n_train": int(train_rows.size),
            "n_val": int(val_rows.size),
        }

    guard = {
        "stage": stage,
        "ledger_lineage_stage": effective_lineage_stage,
        "datasets": datasets,
        "ledger": ledger,
        "ledger_sha256": ledger_sha,
        "ledger_name": ledger["ledger_name"],
        "ledger_version": lineage_contract["ledger_version"],
        "roles": roles,
    }
    verify_stage_indices(pools, guard, train_idx, val_idx)
    manifest = {
        # Compatibility keys retained for existing split-manifest readers; the
        # actual rows come from the ledger, never from a runtime resplit.
        "seed": split_source["seed"],
        "val_fraction": split_source["val_fraction"],
        "split_policy": ledger["ledger_name"],
        "ledger_schema_version": ledger["schema_version"],
        "ledger_name": ledger["ledger_name"],
        "ledger_version": lineage_contract["ledger_version"],
        "ledger_lineage_stage": effective_lineage_stage,
        "ledger_sha256": ledger_sha,
        "parent_ledger_sha256": ledger.get("parent_ledger_sha256"),
        "ledger_seed": ledger.get("seed"),
        "stage": stage,
        "datasets": manifest_datasets,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }
    return {
        "pools": pools,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "manifest": manifest,
        "guard": guard,
    }


def inject_guard_test_row(
    pools: list[Any],
    guard: dict[str, Any],
    train_idx: list[tuple[int, int]],
    val_idx: list[tuple[int, int]],
    spec: str,
) -> None:
    """Inject a row in memory so a CLI dry run can prove rejection."""
    parts = spec.split(":")
    require(len(parts) in {3, 4},
            "guard test must be DATASET:train|val:ROLE[:BLOCK]")
    dataset, lane, role = parts[:3]
    require(dataset in guard["datasets"], f"guard test dataset not selected: {dataset}")
    require(lane in {"train", "val"}, f"guard test lane must be train or val: {lane}")
    require(role in REQUIRED_ROLES, f"guard test has unknown role: {role}")
    blocks = guard["roles"][dataset][role]
    require(blocks, f"guard test role is empty: {dataset}/{role}")
    block = parts[3] if len(parts) == 4 else blocks[0]
    require(
        block in blocks,
        f"guard test block {block!r} is not assigned to {dataset}/{role}",
    )
    pool_i = next(i for i, pool in enumerate(pools) if pool.name == dataset)
    rows = np.flatnonzero(np.asarray(pools[pool_i].block).astype(str) == block)
    require(rows.size > 0, f"guard test block has no rows: {dataset}/{block}")
    target = train_idx if lane == "train" else val_idx
    target.append((pool_i, int(rows[0])))
    print(f"guard-test injected {dataset} row {int(rows[0])} block={block!r} "
          f"into {guard['stage']}_{lane} from role={role}", flush=True)
    verify_stage_indices(pools, guard, train_idx, val_idx)


def print_guard_summary(result: dict[str, Any]) -> None:
    manifest = result["manifest"]
    print(f"ledger guard PASS stage={manifest['stage']} "
          f"lineage_stage={manifest['ledger_lineage_stage']} "
          f"ledger={manifest['ledger_name']} version={manifest['ledger_version']} "
          f"schema={manifest['ledger_schema_version']} "
          f"sha256={manifest['ledger_sha256']}")
    for dataset, entry in manifest["datasets"].items():
        print(f"  {dataset}: train={len(entry['train_blocks'])} blocks/"
              f"{entry['n_train']} rows val={len(entry['val_blocks'])} blocks/"
              f"{entry['n_val']} rows")
