from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return normalized * self.weight.to(x.dtype)


class VibrationToGemmaProjector(nn.Module):
    """Maps modality tokens into Gemma's main token-embedding space."""

    def __init__(
        self,
        input_width: int,
        gemma_hidden_size: int,
        hidden_multiplier: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        hidden = max(input_width, gemma_hidden_size) * hidden_multiplier
        self.input_norm = RMSNorm(input_width)
        self.mlp = nn.Sequential(
            nn.Linear(input_width, hidden, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, gemma_hidden_size, bias=False),
        )
        self.output_norm = RMSNorm(gemma_hidden_size)
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(tokens)
        x = self.mlp(x)
        x = self.output_norm(x)
        return x * self.log_scale.exp().to(x.dtype)
