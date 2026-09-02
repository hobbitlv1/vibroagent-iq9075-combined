from __future__ import annotations

import torch
from torch import nn

from .encoder import PhysicsFeatureEncoder



class CommonModeResidualAdapter(nn.Module):
    """Zero-gated target-minus-network-common correction for physics tokens.

    The adapter is deliberately not a classifier. It receives only numeric physics
    features, computes a robust target-network common mode, and emits a continuous
    correction for the existing physics-token slots. At initialization the gate is
    exactly zero, so enabling the module reproduces the immutable champion exactly.
    """

    def __init__(
        self,
        *,
        feature_count: int,
        width: int,
        physics_tokens_per_board: int,
        attention_heads: int = 4,
        transformer_layers: int = 1,
        dropout: float = 0.05,
        minimum_common_targets: int = 3,
        maximum_gate: float = 0.5,
        residual_clip: float = 8.0,
        scale_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        if feature_count < 1:
            raise ValueError("feature_count must be positive")
        if physics_tokens_per_board < 1:
            raise ValueError("physics_tokens_per_board must be positive")
        if not 1 <= minimum_common_targets <= 5:
            raise ValueError("minimum_common_targets must be in [1, 5]")
        if maximum_gate <= 0:
            raise ValueError("maximum_gate must be positive")
        if residual_clip <= 0:
            raise ValueError("residual_clip must be positive")
        if scale_floor <= 0:
            raise ValueError("scale_floor must be positive")

        self.feature_count = int(feature_count)
        self.width = int(width)
        self.physics_tokens_per_board = int(physics_tokens_per_board)
        self.minimum_common_targets = int(minimum_common_targets)
        self.maximum_gate = float(maximum_gate)
        self.residual_clip = float(residual_clip)
        self.scale_floor = float(scale_floor)

        self.encoder = PhysicsFeatureEncoder(
            feature_count=self.feature_count,
            width=self.width,
            output_tokens=self.physics_tokens_per_board,
            attention_heads=attention_heads,
            transformer_layers=transformer_layers,
            dropout=dropout,
        )
        self.output_norm = nn.LayerNorm(self.width)
        self.output_projection = nn.Linear(self.width, self.width, bias=False)
        nn.init.eye_(self.output_projection.weight)
        # One independently learned gate for each retained physics-token slot.
        # tanh(0) == 0 exactly, preserving the champion before training.
        self.gate_logits = nn.Parameter(torch.zeros(self.physics_tokens_per_board))

    @property
    def gate_values(self) -> torch.Tensor:
        return torch.tanh(self.gate_logits) * self.maximum_gate

    def residual_inputs(
        self,
        physics_features: torch.Tensor,
        physics_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if physics_features.ndim != 3 or physics_features.shape[1] != 6:
            raise ValueError(
                "physics_features must have shape [batch, 6, features], got "
                f"{tuple(physics_features.shape)}"
            )
        if physics_mask.shape != physics_features.shape:
            raise ValueError("physics feature/mask shape mismatch")
        if physics_features.shape[-1] != self.feature_count:
            raise ValueError(
                f"expected {self.feature_count} physics features, got {physics_features.shape[-1]}"
            )

        mask = physics_mask.to(torch.bool)
        values = torch.nan_to_num(physics_features, nan=0.0, posinf=0.0, neginf=0.0)
        targets = values[:, 1:, :]
        target_mask = mask[:, 1:, :]

        valid_count = target_mask.sum(dim=1)
        common_mask = valid_count >= self.minimum_common_targets
        sentinel = torch.full_like(targets, torch.finfo(targets.dtype).max)
        sorted_targets = torch.topk(
            torch.where(target_mask, targets, sentinel),
            k=3,
            dim=1,
            largest=False,
            sorted=True,
        ).values
        common_for_three = sorted_targets[:, 1, :]
        common_for_four = 0.5 * (sorted_targets[:, 1, :] + sorted_targets[:, 2, :])
        common_for_five = sorted_targets[:, 2, :]
        common = torch.where(
            valid_count >= 5,
            common_for_five,
            torch.where(valid_count == 4, common_for_four, common_for_three),
        )
        common = torch.where(common_mask, common, torch.zeros_like(common))

        absolute_deviation = torch.abs(targets - common.unsqueeze(1))
        sorted_deviation = torch.topk(
            torch.where(target_mask & common_mask.unsqueeze(1), absolute_deviation, sentinel),
            k=3,
            dim=1,
            largest=False,
            sorted=True,
        ).values
        scale_for_three = sorted_deviation[:, 1, :]
        scale_for_four = 0.5 * (sorted_deviation[:, 1, :] + sorted_deviation[:, 2, :])
        scale_for_five = sorted_deviation[:, 2, :]
        scale = torch.where(
            valid_count >= 5,
            scale_for_five,
            torch.where(valid_count == 4, scale_for_four, scale_for_three),
        )
        scale = torch.where(common_mask, scale, torch.zeros_like(scale))
        # The magnitude floor prevents a single almost-identical target family from
        # creating arbitrarily large residuals while preserving local deviations.
        scale = torch.maximum(scale, torch.full_like(scale, self.scale_floor))

        residual = (values - common.unsqueeze(1)) / scale.unsqueeze(1)
        residual = torch.clamp(residual, -self.residual_clip, self.residual_clip)
        residual_mask = mask & common_mask.unsqueeze(1)

        # The common mode is defined by the five target boards. The reference board
        # is intentionally not corrected; its original baseline token remains intact.
        residual[:, 0, :] = 0.0
        residual_mask[:, 0, :] = False
        residual = residual * residual_mask.to(residual.dtype)
        return residual, residual_mask, common_mask

    def forward(
        self,
        physics_features: torch.Tensor,
        physics_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual, residual_mask, _ = self.residual_inputs(physics_features, physics_mask)
        batch, boards, features = residual.shape
        flat_values = residual.reshape(batch * boards, features)
        flat_mask = residual_mask.reshape(batch * boards, features)
        tokens, token_mask, _ = self.encoder(flat_values, flat_mask)
        tokens = self.output_projection(self.output_norm(tokens))
        tokens = tokens.reshape(
            batch,
            boards,
            self.physics_tokens_per_board,
            self.width,
        )
        token_mask = token_mask.reshape(batch, boards, self.physics_tokens_per_board)
        gate = self.gate_values.to(tokens.dtype).view(1, 1, -1, 1)
        correction = tokens * gate
        correction = correction * token_mask.unsqueeze(-1).to(tokens.dtype)
        return correction
