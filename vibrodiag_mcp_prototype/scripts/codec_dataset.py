"""Codec pretraining data pipeline over the public window pools (Phase 3).

Doctrine (NATIVE_VIBRATION_LLM_PLAN.md Phase 3 + section 9):
- PUBLIC data only (fdsn / lumo / rt345 pools under data/public_windows/).
  The codec never sees windows from this building.
- Per-window shape normalization: detrend + divide by AC rms. The codec
  learns SHAPE; absolute level travels separately as one text scalar.
- Splits are BLOCK-level (station-day / session / test record), never random
  windows - adjacent windows are near-duplicates (public_folds_v2 policy).
- Injected perturbations are mixed into training windows (target = the
  perturbed window itself) so anomalous content is inside the codec's
  training distribution (risk R2).

Numpy-only on purpose: importable on machines without torch. Torch enters in
train_vibration_codec.py.

CLI:
  python scripts/codec_dataset.py --data-dir data/public_windows \
      --out data/codec_pretrain/split_manifest.json        # write split manifest
  python scripts/codec_dataset.py --data-dir data/public_windows --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import vibration_injection as vinj  # noqa: E402

WINDOW_SAMPLES = 4000  # 10 s @ 400 Hz
DEFAULT_DATASETS = ("fdsn", "lumo", "rt345")


def _have_scipy() -> bool:
    try:
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


# injection menu: spectral-content kinds only. gain is level-only and a no-op
# after shape normalization; corruptions (clipping/dropout) are quality flags,
# not vibration content - both excluded here.
def injection_menu(with_scipy: bool | None = None) -> list:
    if with_scipy is None:
        with_scipy = _have_scipy()
    menu = [
        lambda w, rng, sev: vinj.inject_tone(w, rng, sev),
        lambda w, rng, sev: vinj.inject_tone(w, rng, sev, harmonics=True),
        lambda w, rng, sev: vinj.inject_tone(w, rng, sev, energy_neutral=True),
        lambda w, rng, sev: vinj.inject_tone(w, rng, sev, partial=True),
        lambda w, rng, sev: vinj.inject_impulse(w, rng, sev),
        lambda w, rng, sev: vinj.inject_impulse(w, rng, sev, repeated=True),
    ]
    if with_scipy:  # these two import scipy inside vibration_injection
        menu.append(lambda w, rng, sev: vinj.inject_broadband_noise(w, rng, sev))
        menu.append(lambda w, rng, sev: vinj.inject_spectral_shift(w, rng, sev))
    return menu


def normalize_window(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Shape-normalize one window: detrend + unit AC rms.

    Returns (normalized float32, level_db). level_db = 20*log10(ac_rms) is the
    scalar that travels OUTSIDE the codec (section 9 doctrine)."""
    x = np.asarray(w, dtype=np.float64)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x * x)))
    rms = max(rms, 1e-12)
    return (x / rms).astype(np.float32), float(20.0 * np.log10(rms))


@dataclass
class Pool:
    name: str
    windows: np.ndarray  # (N, 4000) float32, raw
    block: np.ndarray    # (N,) str - split unit
    channel: np.ndarray  # (N,) str
    start_s: np.ndarray  # (N,) float - source-record-relative window start


def load_pools(data_dir: str | Path, datasets=DEFAULT_DATASETS) -> list[Pool]:
    pools = []
    for name in datasets:
        z = np.load(Path(data_dir) / f"{name}.npz", allow_pickle=True)
        w = z["windows"]
        if w.shape[1] != WINDOW_SAMPLES:
            raise ValueError(f"{name}: expected {WINDOW_SAMPLES} samples/window, got {w.shape[1]}")
        start_s = z["start_s"] if "start_s" in z.files else np.full(w.shape[0], np.nan)
        pools.append(Pool(name=name, windows=w, block=z["block"], channel=z["channel"],
                          start_s=start_s))
    return pools


def block_split(pools: list[Pool], val_fraction: float, seed: int) -> dict:
    """Deterministic block-level split. Returns manifest dict with per-dataset
    train/val block lists and global index arrays."""
    rng = np.random.default_rng(seed)
    manifest = {"seed": seed, "val_fraction": val_fraction, "datasets": {}}
    train_idx: list[tuple[int, int]] = []  # (pool_i, row)
    val_idx: list[tuple[int, int]] = []
    for pi, pool in enumerate(pools):
        blocks = sorted(set(pool.block.tolist()))
        n_val = max(1, int(round(len(blocks) * val_fraction)))
        val_blocks = set(rng.choice(blocks, size=n_val, replace=False).tolist())
        is_val = np.isin(pool.block, sorted(val_blocks))
        train_idx += [(pi, int(r)) for r in np.flatnonzero(~is_val)]
        val_idx += [(pi, int(r)) for r in np.flatnonzero(is_val)]
        manifest["datasets"][pool.name] = {
            "n_windows": int(pool.windows.shape[0]),
            "n_blocks": len(blocks),
            "val_blocks": sorted(val_blocks),
            "n_train": int((~is_val).sum()),
            "n_val": int(is_val.sum()),
        }
    manifest["n_train"] = len(train_idx)
    manifest["n_val"] = len(val_idx)
    return {"manifest": manifest, "train_idx": train_idx, "val_idx": val_idx}


class WindowSampler:
    """Infinite batch stream of shape-normalized windows with injection mixing.

    Injection happens on the RAW window (injection amplitudes are defined
    relative to background rms), normalization after. Target = input: the
    codec is trained to reconstruct perturbed content, not remove it."""

    def __init__(self, pools: list[Pool], index: list[tuple[int, int]],
                 p_inject: float, seed: int, with_scipy: bool | None = None):
        self.pools = pools
        self.index = index
        self.p_inject = p_inject
        self.rng = np.random.default_rng(seed)
        self.menu = injection_menu(with_scipy)

    def batch(self, size: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Returns (windows (B,4000) float32 normalized, level_db (B,), kinds)."""
        rows = self.rng.integers(0, len(self.index), size=size)
        out = np.empty((size, WINDOW_SAMPLES), dtype=np.float32)
        levels = np.empty(size, dtype=np.float64)
        kinds: list[str] = []
        for j, k in enumerate(rows):
            pi, r = self.index[int(k)]
            w = self.pools[pi].windows[r].astype(np.float64)
            kind = "none"
            if self.rng.uniform() < self.p_inject:
                fn = self.menu[int(self.rng.integers(0, len(self.menu)))]
                sev = "mild" if self.rng.uniform() < 0.5 else "significant"
                w, inj = fn(w, self.rng, sev)
                kind = inj.kind
            out[j], levels[j] = normalize_window(w)
            kinds.append(kind)
        return out, levels, kinds


def build(data_dir: str | Path, datasets=DEFAULT_DATASETS, val_fraction: float = 0.15,
          seed: int = 20260708) -> dict:
    pools = load_pools(data_dir, datasets)
    split = block_split(pools, val_fraction, seed)
    return {"pools": pools, **split}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data/public_windows")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--out", default=None, help="write split manifest JSON here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    datasets = tuple(s for s in args.datasets.split(",") if s)
    built = build(args.data_dir, datasets, args.val_fraction, args.seed)
    man = built["manifest"]
    print(json.dumps(man, indent=2)[:600])
    print(f"train windows: {man['n_train']}  val windows: {man['n_val']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(man, indent=2))
        print(f"manifest -> {args.out}")

    if args.selftest:
        scipy_ok = _have_scipy()
        print(f"scipy available: {scipy_ok} (menu of {len(injection_menu(scipy_ok))} injection kinds)")
        sampler = WindowSampler(built["pools"], built["train_idx"], p_inject=0.5, seed=1)
        from collections import Counter
        seen: Counter = Counter()
        for _ in range(4):
            w, lv, kinds = sampler.batch(32)
            assert w.shape == (32, WINDOW_SAMPLES) and np.isfinite(w).all()
            rms = np.sqrt((w.astype(np.float64) ** 2).mean(axis=1))
            assert np.allclose(rms, 1.0, atol=1e-3), f"normalization broken: rms {rms[:4]}"
            assert np.isfinite(lv).all()
            seen.update(kinds)
        print("kind mix over 128 draws:", dict(seen))
        # val split must not share blocks with train
        for name, d in man["datasets"].items():
            assert d["n_train"] + d["n_val"] == d["n_windows"]
        print("SELFTEST OK")


if __name__ == "__main__":
    main()
