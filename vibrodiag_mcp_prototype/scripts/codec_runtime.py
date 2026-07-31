"""Frozen codec-v1 inference architecture and waveform normalization.

This module contains only what the live codes_v3 path needs to load the
hash-pinned codec checkpoint and encode 10-second, 400 Hz windows. It has no
training CLI, dataset mutation, or experimental version-selection behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

WINDOW_SAMPLES = 4000

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



def normalize_window(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Shape-normalize one window: detrend + unit AC rms.

    Returns (normalized_float32, level_db). The absolute level remains
    separate from the codec's shape codes."""
    x = np.asarray(w, dtype=np.float64)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x * x)))
    rms = max(rms, 1e-12)
    return (x / rms).astype(np.float32), float(20.0 * np.log10(rms))


__all__ = ["CodecConfig", "RVQCodec", "normalize_window"]
