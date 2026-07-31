"""Codec-v2 trainer - residual-VQ conv autoencoder on public windows.

Plan (NATIVE_VIBRATION_LLM_PLAN.md, Phase 3): conv encoder, 32x downsample
(400 Hz -> 12.5 frames/s -> 125 frames per 10 s window), residual VQ with
2 codebooks x 1024 entries (250 codes per window), conv decoder, ~1-4 M params.
Trained self-supervised (reconstruction + commitment) on shape-normalized
public windows with injected perturbations mixed in (risk R2).

Requires torch + scipy (GPU strongly recommended). Data: codec_dataset.py.

Production (the frozen config supplies every training degree of freedom):

  python scripts/train_vibration_codec.py \
      --run-config configs/codec_v2_train.json \
      --out-dir runs/codec_v2 --device cuda

The v2 block-role ledger is authoritative by default. Use
``--ledger-guard-check`` to validate pool hashes and the exact codec train/val
rows without creating an output directory or initializing a model.

Definitive accept/reject is scripts/codec_v2_gate.py, not the loss curve.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import queue
import random
import shlex
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent))
from block_role_guard import (  # noqa: E402
    AUTHORITATIVE_DATASETS,
    AUTHORITATIVE_LEDGER_SHA256,
    AUTHORITATIVE_NPZ_SHA256,
    guarded_stage_split,
    inject_guard_test_row,
    print_guard_summary,
    require,
    sha256_file,
    verify_authoritative_inputs,
    verify_stage_indices,
)
from codec_dataset import (  # noqa: E402
    WINDOW_SAMPLES,
    WindowSampler,
    injection_menu,
    load_pools,
)


@dataclass
class CodecConfig:
    window_samples: int = WINDOW_SAMPLES
    fs_hz: float = 400.0
    base_channels: int = 32
    max_channels: int = 256
    latent_dim: int = 128
    n_downsample: int = 5              # 2^5 = 32x -> 12.5 frames/s
    codebooks: int = 2
    codebook_size: int = 1024
    ema_decay: float = 0.99
    ema_eps: float = 1e-5
    commit_weight: float = 0.25
    stft_sizes: tuple = (256, 512, 1024)
    reseed_every: int = 500            # dead-code reseed cadence (steps)
    reseed_min_use: float = 0.5        # EMA cluster size below this = dead


CODEC_VERSION = "codec-v2"
RUN_CONFIG_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = "codec-v2-checkpoint-v1"
DATA_BATCH_SEED_STRIDE = 1_000_003


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_frozen_run_config(path: str | Path) -> tuple[dict[str, Any], CodecConfig]:
    config_path = Path(path)
    require(config_path.is_file(), f"run config not found: {config_path}")
    try:
        doc = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"fail-closed: cannot read run config {config_path}: {exc}") from exc
    require(isinstance(doc, dict), "run config root must be an object")
    require(doc.get("schema_version") == RUN_CONFIG_SCHEMA_VERSION,
            f"run config schema must be {RUN_CONFIG_SCHEMA_VERSION}")
    require(doc.get("codec_version") == CODEC_VERSION,
            f"run config codec_version must be {CODEC_VERSION}")
    integrity = doc.get("integrity")
    require(isinstance(integrity, dict), "run config missing integrity object")
    require(integrity.get("require_input_manifest_for_training") is True,
            "production run config must require the immutable INPUT manifest")
    require(integrity.get("ledger_sha256") == AUTHORITATIVE_LEDGER_SHA256,
            "run config ledger pin differs from authoritative ledger v2")
    require(integrity.get("npz_sha256") == AUTHORITATIVE_NPZ_SHA256,
            "run config NPZ pins differ from authoritative public pools")
    data = doc.get("data")
    require(isinstance(data, dict), "run config missing data object")
    require(tuple(data.get("datasets", ())) == AUTHORITATIVE_DATASETS,
            "run config datasets must be the complete authoritative set")
    require(data.get("stage") == "codec", "run config data.stage must be codec")
    require(data.get("public_data_only") is True,
            "run config must declare public_data_only=true")
    architecture = doc.get("architecture")
    require(isinstance(architecture, dict), "run config missing architecture object")
    expected_fields = set(CodecConfig.__dataclass_fields__)
    require(set(architecture) == expected_fields,
            "run config architecture keys differ from CodecConfig: "
            f"missing={sorted(expected_fields - set(architecture))}, "
            f"extra={sorted(set(architecture) - expected_fields)}")
    converted = {
        key: tuple(value) if key == "stft_sizes" else value
        for key, value in architecture.items()
    }
    model_config = CodecConfig(**converted)
    require(model_config.codebooks == 2 and model_config.codebook_size == 1024,
            "codec-v2 must expose exactly 2 x 1024 = 2048 code indices")
    training = doc.get("training")
    require(isinstance(training, dict), "run config missing training object")
    required_training = {
        "steps", "batch_size", "learning_rate", "warmup_steps",
        "injection_probability", "seed", "workers", "eval_every",
        "checkpoint_every", "validation_windows_per_lane", "weight_decay",
        "gradient_clip_norm", "checkpoint_selection", "fresh_start_required",
        "data_batch_seed_stride",
    }
    require(set(training) == required_training,
            "run config training keys differ from the frozen schema: "
            f"missing={sorted(required_training - set(training))}, "
            f"extra={sorted(set(training) - required_training)}")
    require(training["checkpoint_selection"] ==
            "minimum_mean_val_clean_val_injected_multires_stft",
            "unsupported checkpoint-selection rule")
    require(training["fresh_start_required"] is True,
            "codec-v2 must train from a fresh output directory")
    require(training["data_batch_seed_stride"] == DATA_BATCH_SEED_STRIDE,
            "run config data-batch seed stride changed")
    objective = doc.get("objective")
    require(objective == {
        "waveform_l1_weight": 1.0,
        "multires_log_stft_weight": 1.0,
        "commitment_weight_source": "architecture.commit_weight",
        "target": "input_window_after_registered_spectral_augmentation",
    }, "run config objective differs from the frozen codec-v2 objective")
    return doc, model_config


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_record() -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name) for name in ("numpy", "scipy", "torch")
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())],
        },
    }


class ResUnit(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(), nn.Conv1d(ch, ch, 3, padding=1),
            nn.ELU(), nn.Conv1d(ch, ch, 1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, cfg: CodecConfig):
        super().__init__()
        ch = cfg.base_channels
        layers: list[nn.Module] = [nn.Conv1d(1, ch, 7, padding=3)]
        for _ in range(cfg.n_downsample):
            nxt = min(ch * 2, cfg.max_channels)
            layers += [ResUnit(ch), nn.ELU(), nn.Conv1d(ch, nxt, 4, stride=2, padding=1)]
            ch = nxt
        layers += [ResUnit(ch), nn.ELU(), nn.Conv1d(ch, cfg.latent_dim, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):              # (B,1,T) -> (B,D,T/32)
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, cfg: CodecConfig):
        super().__init__()
        chs = [cfg.base_channels]
        for _ in range(cfg.n_downsample):
            chs.append(min(chs[-1] * 2, cfg.max_channels))
        chs = chs[::-1]                # e.g. [256,256,256,128,64,32]
        layers: list[nn.Module] = [nn.Conv1d(cfg.latent_dim, chs[0], 3, padding=1)]
        for i in range(cfg.n_downsample):
            layers += [ResUnit(chs[i]), nn.ELU(),
                       nn.ConvTranspose1d(chs[i], chs[i + 1], 4, stride=2, padding=1)]
        layers += [ResUnit(chs[-1]), nn.ELU(), nn.Conv1d(chs[-1], 1, 7, padding=3)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):              # (B,D,T/32) -> (B,1,T)
        return self.net(z)


class VQEma(nn.Module):
    """One EMA-updated codebook (van den Oord style)."""

    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.cfg = cfg
        embed = torch.randn(cfg.codebook_size, cfg.latent_dim) * 0.1
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("cluster_size", torch.zeros(cfg.codebook_size))

    def forward(self, x):              # x: (N, D) float32
        dist = (x.pow(2).sum(1, keepdim=True)
                - 2 * x @ self.embed.t()
                + self.embed.pow(2).sum(1))
        idx = dist.argmin(1)
        quant = self.embed[idx]
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.cfg.codebook_size).to(x.dtype)
                counts = onehot.sum(0)
                embed_sum = onehot.t() @ x
                d = self.cfg.ema_decay
                self.cluster_size.mul_(d).add_(counts, alpha=1 - d)
                self.embed_avg.mul_(d).add_(embed_sum, alpha=1 - d)
                n = self.cluster_size.sum()
                smoothed = ((self.cluster_size + self.cfg.ema_eps)
                            / (n + self.cfg.codebook_size * self.cfg.ema_eps) * n)
                self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))
        return quant, idx

    @torch.no_grad()
    def reseed_dead(self, batch_vecs: torch.Tensor) -> int:
        dead = self.cluster_size < self.cfg.reseed_min_use
        n_dead = int(dead.sum())
        if n_dead == 0 or batch_vecs.shape[0] == 0:
            return 0
        pick = batch_vecs[torch.randint(0, batch_vecs.shape[0], (n_dead,),
                                        device=batch_vecs.device)]
        self.embed[dead] = pick
        self.embed_avg[dead] = pick
        self.cluster_size[dead] = 1.0
        return n_dead


class RVQCodec(nn.Module):
    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.quantizers = nn.ModuleList(VQEma(cfg) for _ in range(cfg.codebooks))

    def quantize(self, z):             # z: (B,D,T') -> (quant_st, commit, codes)
        b, d, t = z.shape
        # fp32 for codebook distances/EMA even under bf16 autocast
        flat = z.permute(0, 2, 1).reshape(-1, d).float()
        residual = flat
        quant_total = torch.zeros_like(flat)
        commit = flat.new_zeros(())
        codes = []
        for q in self.quantizers:
            quant_i, idx = q(residual)
            commit = commit + F.mse_loss(residual, quant_i.detach())
            quant_total = quant_total + quant_i
            residual = residual - quant_i.detach()
            codes.append(idx.view(b, t))
        quant_st = flat + (quant_total - flat).detach()   # straight-through
        return quant_st.view(b, t, d).permute(0, 2, 1), commit, torch.stack(codes, 1)

    def forward(self, x):              # x: (B,1,T)
        z = self.encoder(x)
        quant, commit, codes = self.quantize(z)
        recon = self.decoder(quant)
        return recon, commit, codes, z

    @torch.no_grad()
    def encode_codes(self, x):
        z = self.encoder(x)
        _, _, codes = self.quantize(z)
        return codes                   # (B, codebooks, T')

    @torch.no_grad()
    def reconstruct(self, x):
        recon, _, _, _ = self.forward(x)
        return recon


def multires_stft_loss(recon: torch.Tensor, target: torch.Tensor,
                       sizes) -> torch.Tensor:
    """L1 on log magnitudes across resolutions; fp32 for STFT stability."""
    r = recon.squeeze(1).float()
    t = target.squeeze(1).float()
    loss = r.new_zeros(())
    for n_fft in sizes:
        win = torch.hann_window(n_fft, device=r.device)
        sr = torch.stft(r, n_fft, hop_length=n_fft // 4, window=win,
                        return_complex=True).abs()
        st = torch.stft(t, n_fft, hop_length=n_fft // 4, window=win,
                        return_complex=True).abs()
        loss = loss + F.l1_loss(torch.log(sr + 1e-5), torch.log(st + 1e-5))
    return loss / len(sizes)


class DeterministicPrefetcher:
    """Ordered prefetch with one independent RNG seed per global step.

    Worker scheduling cannot change which augmentation batch a step consumes.
    The complete data RNG state is therefore the frozen base seed, stride, and
    next global step, which is also written into every checkpoint.
    """

    def __init__(self, pools, index, p_inject, batch_size, workers, seed, steps):
        self.pools = pools
        self.index = index
        self.p_inject = p_inject
        self.batch_size = batch_size
        self.workers = max(0, int(workers))
        self.seed = int(seed)
        self.steps = int(steps)
        self.next_expected = 0
        self.executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
            if self.workers else None
        )
        self.pending: dict[int, concurrent.futures.Future] = {}
        self.depth = max(1, self.workers * 2)
        if self.executor:
            self._fill(0)

    def batch_seed(self, step: int) -> int:
        return self.seed + DATA_BATCH_SEED_STRIDE * int(step)

    def _produce(self, step: int) -> np.ndarray:
        sampler = WindowSampler(
            self.pools,
            self.index,
            p_inject=self.p_inject,
            seed=self.batch_seed(step),
            with_scipy=True,
        )
        windows, _, _ = sampler.batch(self.batch_size)
        return windows

    def _fill(self, start: int) -> None:
        if not self.executor:
            return
        for step in range(start, min(self.steps, start + self.depth)):
            if step not in self.pending:
                self.pending[step] = self.executor.submit(self._produce, step)

    def next(self, step: int) -> np.ndarray:
        require(step == self.next_expected,
                f"data prefetch requested step {step}, expected {self.next_expected}")
        if self.executor:
            self._fill(step)
            future = self.pending.pop(step)
            windows = future.result()
            self._fill(step + 1)
        else:
            windows = self._produce(step)
        self.next_expected += 1
        return windows

    def state(self) -> dict[str, Any]:
        return {
            "scheme": "per_global_step_numpy_default_rng_v1",
            "base_seed": self.seed,
            "stride": DATA_BATCH_SEED_STRIDE,
            "next_step": self.next_expected,
        }

    def close(self) -> None:
        if self.executor:
            for future in self.pending.values():
                future.cancel()
            self.executor.shutdown(wait=True, cancel_futures=True)


def fixed_val_batches(pools, val_idx, seed, n_each=384):
    """Two frozen val sets: clean and injected (p=1)."""
    clean = WindowSampler(pools, val_idx, p_inject=0.0, seed=seed)
    inj = WindowSampler(pools, val_idx, p_inject=1.0, seed=seed + 1)
    wc, _, _ = clean.batch(n_each)
    wi, _, _ = inj.batch(n_each)
    return wc, wi


@torch.no_grad()
def evaluate(model, batches, device, chunk=64):
    model.eval()
    out = {}
    for name, arr in batches.items():
        tot, n = 0.0, 0
        for i in range(0, len(arr), chunk):
            x = torch.from_numpy(arr[i:i + chunk]).unsqueeze(1).to(device)
            recon = model.reconstruct(x)
            tot += float(multires_stft_loss(recon, x, model.cfg.stft_sizes)) * x.shape[0]
            n += x.shape[0]
        out[name] = tot / max(n, 1)
    model.train()
    return out


def codebook_stats(model) -> list[dict]:
    stats = []
    for q in model.quantizers:
        cs = q.cluster_size.detach().float().cpu().numpy()
        p = cs / max(cs.sum(), 1e-12)
        nz = p[p > 0]
        entropy_bits = float(-(nz * np.log2(nz)).sum())
        stats.append({"entropy_bits": round(entropy_bits, 2),
                      "dead_frac": round(float((cs < 0.5).mean()), 4)})
    return stats


def capture_rng_state(prefetch: DeterministicPrefetcher) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "data_rng": prefetch.state(),
    }


def _flag_present(flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-config", default=None,
                    help="frozen codec-v2 JSON config; production runs must use this")
    ap.add_argument("--data-dir", default="data/public_windows")
    ap.add_argument("--datasets", default="fdsn,lumo,rt345")
    ap.add_argument("--out-dir", default=None,
                    help="training output directory (required unless --ledger-guard-check)")
    ap.add_argument("--block-role-ledger",
                    default="data/audits/block_role_ledger_v2.json",
                    help="authoritative v2 block-role ledger")
    ap.add_argument("--ledger-guard-check", action="store_true",
                    help="validate hashes/roles and exit before output/model creation")
    ap.add_argument("--guard-test-inject", default=None,
                    metavar="DATASET:train|val:ROLE[:BLOCK]",
                    help="guard-check-only in-memory violation used to prove rejection")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--p-inject", type=float, default=0.35)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--checkpoint-every", type=int, default=5000)
    ap.add_argument("--val-windows", type=int, default=384,
                    help="fixed clean and injected validation windows per lane")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny architecture for entry-point mechanics only; never promotable")
    ap.add_argument("--overwrite", action="store_true",
                    help="deprecated unsafe option; codec-v2 always requires a fresh out-dir")
    args = ap.parse_args()

    require(not args.overwrite,
            "--overwrite is forbidden for codec-v2 evidence; choose a fresh --out-dir")
    require(args.guard_test_inject is None or args.ledger_guard_check,
            "--guard-test-inject is permitted only with --ledger-guard-check")
    require(not (args.run_config and args.smoke),
            "--smoke cannot weaken a frozen production --run-config")

    frozen_config: dict[str, Any] | None = None
    frozen_config_sha: str | None = None
    if args.run_config:
        controlled_flags = {
            "--data-dir", "--datasets", "--block-role-ledger", "--steps",
            "--batch-size", "--lr", "--warmup", "--p-inject", "--val-fraction",
            "--seed", "--workers", "--eval-every", "--checkpoint-every",
            "--val-windows", "--weight-decay", "--grad-clip",
        }
        supplied = sorted(flag for flag in controlled_flags if _flag_present(flag))
        require(not supplied,
                "frozen --run-config cannot be combined with controlled CLI overrides: "
                f"{supplied}")
        frozen_config, cfg = load_frozen_run_config(args.run_config)
        frozen_config_sha = sha256_file(args.run_config)
        data_cfg = frozen_config["data"]
        train_cfg = frozen_config["training"]
        args.data_dir = data_cfg["data_dir"]
        args.datasets = ",".join(data_cfg["datasets"])
        args.block_role_ledger = data_cfg["block_role_ledger"]
        args.val_fraction = data_cfg["legacy_val_fraction_argument"]
        args.steps = train_cfg["steps"]
        args.batch_size = train_cfg["batch_size"]
        args.lr = train_cfg["learning_rate"]
        args.warmup = train_cfg["warmup_steps"]
        args.p_inject = train_cfg["injection_probability"]
        args.seed = train_cfg["seed"]
        args.workers = train_cfg["workers"]
        args.eval_every = train_cfg["eval_every"]
        args.checkpoint_every = train_cfg["checkpoint_every"]
        args.val_windows = train_cfg["validation_windows_per_lane"]
        args.weight_decay = train_cfg["weight_decay"]
        args.grad_clip = train_cfg["gradient_clip_norm"]
    else:
        cfg = CodecConfig()

    if args.smoke:
        # Keep the real encoder/RVQ/decoder/loss/checkpoint path while bounding CPU
        # time and memory. Explicit CLI values win; otherwise safe smoke defaults.
        cfg = CodecConfig(base_channels=4, max_channels=16, latent_dim=16,
                          codebook_size=32, stft_sizes=(128, 256), reseed_every=25)
        if not _flag_present("--steps"):
            args.steps = 50
        if not _flag_present("--batch-size"):
            args.batch_size = 2
        if not _flag_present("--workers"):
            args.workers = 0
        if not _flag_present("--eval-every"):
            args.eval_every = 25
        if not _flag_present("--checkpoint-every"):
            args.checkpoint_every = 25
        if not _flag_present("--val-windows"):
            args.val_windows = 6

    require(math.isclose(args.val_fraction, 0.15, abs_tol=1e-12),
            "--val-fraction cannot resplit block_role_ledger_v2; expected 0.15")
    require(args.steps > 0, "--steps must be positive")
    require(args.batch_size > 0, "--batch-size must be positive")
    require(args.workers >= 0, "--workers must be nonnegative")
    require(args.eval_every > 0 and args.checkpoint_every > 0,
            "evaluation/checkpoint cadences must be positive")
    require(args.checkpoint_every % args.eval_every == 0,
            "--checkpoint-every must be a multiple of --eval-every")
    require(args.val_windows > 0, "--val-windows must be positive")
    require(0.0 <= args.p_inject <= 1.0, "--p-inject must be in [0,1]")
    require(args.warmup >= 0, "--warmup must be nonnegative")
    require(args.weight_decay >= 0, "--weight-decay must be nonnegative")
    require(args.grad_clip > 0, "--grad-clip must be positive")
    datasets = tuple(s for s in args.datasets.split(",") if s)

    bundle_root = Path(__file__).resolve().parents[1]
    input_manifest_path = bundle_root / "INPUT_MANIFEST.json"
    input_bundle_receipt: dict[str, Any] | None = None
    if input_manifest_path.is_file():
        try:
            from verify_codec_v2_bundle import verify_bundle
            input_bundle_receipt = verify_bundle(bundle_root)
        except Exception as exc:
            raise SystemExit(
                f"fail-closed: immutable INPUT bundle verification failed: {exc}"
            ) from exc
        print(
            "immutable INPUT bundle PASS "
            f"manifest_sha256={input_bundle_receipt['manifest_sha256']} "
            f"files={input_bundle_receipt['verified_file_count']}"
        )
    elif frozen_config is not None and not args.ledger_guard_check:
        require(False,
                "frozen production training requires bundle-root INPUT_MANIFEST.json")

    # Hash before np.load(... allow_pickle=True). A self-consistent replacement
    # ledger cannot authorize itself because the expected SHA lives in code/config.
    input_integrity = verify_authoritative_inputs(
        args.block_role_ledger, args.data_dir, datasets
    )
    require(len(injection_menu(with_scipy=True)) == 8,
            "the frozen spectral augmentation menu requires scipy and all 8 kinds")
    pools = load_pools(args.data_dir, datasets)
    built = guarded_stage_split(
        pools, args.data_dir, datasets, args.block_role_ledger, stage="codec"
    )
    if args.guard_test_inject:
        inject_guard_test_row(
            built["pools"], built["guard"], built["train_idx"], built["val_idx"],
            args.guard_test_inject,
        )
    print_guard_summary(built)
    if args.ledger_guard_check:
        return
    require(args.out_dir is not None, "--out-dir is required for training")
    if args.device.startswith("cuda"):
        require(torch.cuda.is_available(),
                f"CUDA device requested but torch.cuda.is_available() is false: {args.device}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    existing = sorted(p.name for p in out_dir.iterdir()) if out_dir.exists() else []
    require(not existing,
            f"fresh output directory required; {out_dir} contains {existing[:8]}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = built["manifest"]
    manifest["codec_version"] = CODEC_VERSION
    manifest["training_seed"] = args.seed
    manifest["legacy_val_fraction_argument"] = args.val_fraction
    manifest["input_integrity"] = input_integrity
    atomic_json(out_dir / "split_manifest.json", manifest)

    resolved_config = frozen_config or {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "codec_version": CODEC_VERSION,
        "run_mode": "mechanics_smoke" if args.smoke else "unfrozen_cli_diagnostic",
        "integrity": {
            "ledger_sha256": AUTHORITATIVE_LEDGER_SHA256,
            "npz_sha256": AUTHORITATIVE_NPZ_SHA256,
        },
        "data": {
            "data_dir": args.data_dir,
            "block_role_ledger": args.block_role_ledger,
            "datasets": list(datasets),
            "stage": "codec",
            "public_data_only": True,
            "legacy_val_fraction_argument": args.val_fraction,
        },
        "architecture": asdict(cfg),
        "objective": {
            "waveform_l1_weight": 1.0,
            "multires_log_stft_weight": 1.0,
            "commitment_weight_source": "architecture.commit_weight",
            "target": "input_window_after_registered_spectral_augmentation",
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "warmup_steps": args.warmup,
            "injection_probability": args.p_inject,
            "seed": args.seed,
            "workers": args.workers,
            "eval_every": args.eval_every,
            "checkpoint_every": args.checkpoint_every,
            "validation_windows_per_lane": args.val_windows,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.grad_clip,
            "checkpoint_selection":
                "minimum_mean_val_clean_val_injected_multires_stft",
            "fresh_start_required": True,
            "data_batch_seed_stride": DATA_BATCH_SEED_STRIDE,
        },
    }
    atomic_json(out_dir / "resolved_train_config.json", resolved_config)
    atomic_json(out_dir / "environment.json", environment_record())
    source_files = {
        name: Path(__file__).resolve().parent / name for name in (
            "train_vibration_codec.py", "codec_dataset.py", "block_role_guard.py",
            "vibration_injection.py",
        )
    }
    run_manifest: dict[str, Any] = {
        "schema_version": 1,
        "codec_version": CODEC_VERSION,
        "status": "running",
        "promotion_eligible": not args.smoke and frozen_config is not None,
        "run_mode": "mechanics_smoke" if args.smoke else "production",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "frozen_run_config_sha256": frozen_config_sha,
        "resolved_train_config_sha256": sha256_file(
            out_dir / "resolved_train_config.json"
        ),
        "input_integrity": input_integrity,
        "input_bundle_verification": input_bundle_receipt,
        "source_sha256": {
            name: sha256_file(path) for name, path in source_files.items()
        },
        "checkpoint_selection":
            "minimum_mean_val_clean_val_injected_multires_stft",
    }
    atomic_json(out_dir / "run_manifest.json", run_manifest)

    model = RVQCodec(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    frames = cfg.window_samples // (2 ** cfg.n_downsample)
    print(f"params: {n_params/1e6:.2f} M | {frames} frames x {cfg.codebooks} codebooks "
          f"= {frames * cfg.codebooks} codes / 10 s window")
    print(f"train windows: {manifest['n_train']}  val windows: {manifest['n_val']}  "
          f"device: {args.device}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    # Re-check the concrete rows at the last boundary before either sampler sees
    # them; this catches accidental list mutation in future trainer refactors.
    verify_stage_indices(built["pools"], built["guard"],
                         built["train_idx"], built["val_idx"])
    prefetch = DeterministicPrefetcher(
        built["pools"], built["train_idx"], args.p_inject,
        args.batch_size, args.workers, args.seed, args.steps,
    )
    wc, wi = fixed_val_batches(
        built["pools"], built["val_idx"], args.seed, n_each=args.val_windows
    )
    val_batches = {"val_clean": wc, "val_injected": wi}

    use_bf16 = args.device.startswith("cuda")
    history = open(out_dir / "history.jsonl", "x", encoding="utf-8")
    best = float("inf")
    best_step: int | None = None
    t0 = time.time()
    model.train()
    try:
        for step in range(args.steps):
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            x = torch.from_numpy(prefetch.next(step)).unsqueeze(1).to(args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                recon, commit, _, z = model(x)
            recon_f, x_f = recon.float(), x.float()
            loss = (F.l1_loss(recon_f, x_f)
                    + multires_stft_loss(recon_f, x_f, cfg.stft_sizes)
                    + cfg.commit_weight * commit.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            if (step + 1) % cfg.reseed_every == 0:
                with torch.no_grad():
                    flat = (z.detach().float().permute(0, 2, 1)
                            .reshape(-1, cfg.latent_dim))
                    reseeded = sum(q.reseed_dead(flat) for q in model.quantizers)
                if reseeded:
                    print(f"[{step+1}] reseeded {reseeded} dead codes")

            if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
                val = evaluate(model, val_batches, args.device)
                cb = codebook_stats(model)
                score = (val["val_clean"] + val["val_injected"]) / 2
                rec = {"step": step + 1,
                       "loss": round(float(loss.detach()), 4),
                       "lr": round(lr_at(step), 8),
                       **{k: round(v, 6) for k, v in val.items()},
                       "checkpoint_selection_score": round(score, 6),
                       "codebooks": cb,
                       "sec_per_step": round((time.time() - t0) / (step + 1), 3)}
                history.write(json.dumps(rec, allow_nan=False) + "\n")
                history.flush()
                os.fsync(history.fileno())
                print(rec)
                ckpt = {
                    "checkpoint_schema": CHECKPOINT_SCHEMA,
                    "codec_version": CODEC_VERSION,
                    "config": asdict(cfg),
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": {
                        "name": "linear_warmup_then_cosine_to_0.1x_v1",
                        "step": step + 1,
                        "last_lr": [group["lr"] for group in opt.param_groups],
                        "warmup_steps": args.warmup,
                        "total_steps": args.steps,
                    },
                    "rng_state": capture_rng_state(prefetch),
                    "step": step + 1,
                    "val": val,
                    "split_manifest": manifest,
                    "resolved_train_config_sha256": sha256_file(
                        out_dir / "resolved_train_config.json"
                    ),
                }
                atomic_torch_save(ckpt, out_dir / "last.pt")
                if score < best:
                    best = score
                    best_step = step + 1
                    atomic_torch_save(ckpt, out_dir / "best.pt")
                if ((step + 1) % args.checkpoint_every == 0
                        or step == args.steps - 1):
                    atomic_torch_save(
                        ckpt,
                        out_dir / "checkpoints" / f"step_{step + 1:08d}.pt",
                    )
                if step == args.steps - 1:
                    atomic_torch_save(ckpt, out_dir / "final.pt")
    except BaseException as exc:
        run_manifest["status"] = "failed"
        run_manifest["failure"] = f"{type(exc).__name__}: {exc}"
        run_manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(out_dir / "run_manifest.json", run_manifest)
        raise
    finally:
        prefetch.close()
        history.close()

    run_manifest.update({
        "status": "completed",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "steps_completed": args.steps,
        "best_step": best_step,
        "best_checkpoint_selection_score": best,
        "artifacts": {
            rel: {"sha256": sha256_file(out_dir / rel),
                  "size_bytes": (out_dir / rel).stat().st_size}
            for rel in (
                "best.pt", "last.pt", "final.pt", "history.jsonl",
                "split_manifest.json", "resolved_train_config.json",
                "environment.json",
            )
        },
        "periodic_checkpoints": [
            {"path": path.relative_to(out_dir).as_posix(),
             "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted((out_dir / "checkpoints").glob("step_*.pt"))
        ],
    })
    atomic_json(out_dir / "run_manifest.json", run_manifest)
    print(f"done in {(time.time()-t0)/60:.1f} min; best val score {best:.4f}; "
          f"checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
