from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _safe_padding_mask(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Avoid all-masked rows without tensor-dependent Python control flow."""

    padding = mask.to(torch.bool)
    valid_row = (~padding).any(dim=-1)
    safe_first = padding[..., :1] & valid_row.unsqueeze(-1)
    safe_padding = torch.cat((safe_first, padding[..., 1:]), dim=-1)
    return safe_padding, valid_row


class DepthwiseResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise_in = nn.Conv1d(channels, channels * 2, 1, bias=False)
        self.pointwise_out = nn.Conv1d(channels, channels, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(self.norm(x))
        gate, value = self.pointwise_in(x).chunk(2, dim=1)
        x = self.pointwise_out(F.silu(gate) * value)
        return residual + self.dropout(x)


class ResidualBlock2D(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise_in = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.pointwise_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(self.norm(x))
        gate, value = self.pointwise_in(x).chunk(2, dim=1)
        x = self.pointwise_out(F.silu(gate) * value)
        return residual + self.dropout(x)


class LearnedQueryResampler(nn.Module):
    def __init__(self, width: int, output_tokens: int, attention_heads: int, dropout: float) -> None:
        super().__init__()
        self.output_tokens = int(output_tokens)
        self.queries = nn.Parameter(torch.randn(output_tokens, width) * (width**-0.5))
        self.attention = nn.MultiheadAttention(
            width,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width, bias=False),
        )

    def forward(self, source: torch.Tensor, source_padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        safe_mask, valid_row = _safe_padding_mask(source_padding_mask)
        query = self.queries.unsqueeze(0).expand(source.shape[0], -1, -1)
        attended, _ = self.attention(query, source, source, key_padding_mask=safe_mask, need_weights=False)
        output = self.norm(query + attended)
        output = output + self.feed_forward(output)
        output = output * valid_row[:, None, None].to(output.dtype)
        token_mask = valid_row[:, None].expand(-1, self.output_tokens)
        return output, token_mask


class AxisAwareModalEncoder(nn.Module):
    """Shared 1D temporal encoder for the anti-aliased structural/modal view."""

    def __init__(
        self,
        *,
        width: int,
        depth: int,
        tokens_per_axis: int,
        output_tokens: int,
        kernel_sizes: list[int] | tuple[int, ...],
        dilations: list[int] | tuple[int, ...],
        attention_heads: int,
        fusion_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if depth != len(kernel_sizes) or depth != len(dilations):
            raise ValueError("modal depth must match kernel_sizes and dilations")
        self.width = int(width)
        self.tokens_per_axis = int(tokens_per_axis)
        self.axis_embedding = nn.Embedding(3, width)
        # Per-axis signal + per-sample validity + axis-presence channels.
        self.stem = nn.Sequential(
            nn.Conv1d(3, width, kernel_size=15, stride=2, padding=7, bias=False),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(),
            nn.Conv1d(width, width, kernel_size=9, stride=2, padding=4, bias=False),
        )
        self.blocks = nn.ModuleList(
            [
                DepthwiseResidualBlock1D(width, int(kernel), int(dilation), dropout)
                for kernel, dilation in zip(kernel_sizes, dilations, strict=True)
            ]
        )
        self.final_norm = nn.GroupNorm(_group_count(width), width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=attention_heads,
            dim_feedforward=width * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.axis_fusion = nn.TransformerEncoder(encoder_layer, num_layers=fusion_layers)
        self.resampler = LearnedQueryResampler(width, output_tokens, attention_heads, dropout)

    def forward(
        self,
        values: torch.Tensor,
        axis_mask: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or values.shape[1] != 3:
            raise ValueError(f"modal values must be [batch,3,samples], got {tuple(values.shape)}")
        if axis_mask.shape != values.shape[:2] or sample_mask.shape != (values.shape[0], values.shape[-1]):
            raise ValueError("modal axis/sample mask shape mismatch")
        batch, axes, samples = values.shape
        axis_float = axis_mask.to(values.dtype)
        sample_float = sample_mask.to(values.dtype)
        signal = values * axis_float.unsqueeze(-1) * sample_float.unsqueeze(1)
        flat_signal = signal.reshape(batch * axes, 1, samples)
        flat_sample = sample_float[:, None, :].expand(-1, axes, -1).reshape(batch * axes, 1, samples)
        flat_axis = axis_float.reshape(batch * axes, 1, 1).expand(-1, 1, samples)
        x = torch.cat((flat_signal, flat_sample, flat_axis), dim=1)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = F.silu(self.final_norm(x))
        x = F.adaptive_avg_pool1d(x, self.tokens_per_axis).transpose(1, 2)
        x = x.reshape(batch, axes, self.tokens_per_axis, self.width)
        axis_ids = torch.arange(3, device=values.device)
        x = x + self.axis_embedding(axis_ids).view(1, 3, 1, self.width)
        source = x.reshape(batch, axes * self.tokens_per_axis, self.width)
        source_padding = (~axis_mask).unsqueeze(-1).expand(-1, -1, self.tokens_per_axis).reshape(batch, -1)
        safe_padding, valid_row = _safe_padding_mask(source_padding)
        source = self.axis_fusion(source, src_key_padding_mask=safe_padding)
        source = source * valid_row[:, None, None].to(source.dtype)
        tokens, token_mask = self.resampler(source, source_padding)
        return tokens, token_mask, source


class AxisAwareWidebandEncoder(nn.Module):
    """Numeric complex/log-power STFT encoder that preserves the useful 6 kHz band."""

    def __init__(
        self,
        *,
        input_channels: int,
        width: int,
        depth: int,
        output_tokens: int,
        attention_heads: int,
        fusion_layers: int,
        frequency_patches: int,
        time_patches: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.frequency_patches = int(frequency_patches)
        self.time_patches = int(time_patches)
        self.axis_embedding = nn.Embedding(3, width)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width // 2, 5, stride=(2, 2), padding=2, bias=False),
            nn.GroupNorm(_group_count(width // 2), width // 2),
            nn.SiLU(),
            nn.Conv2d(width // 2, width, 3, stride=(2, 2), padding=1, bias=False),
        )
        self.blocks = nn.ModuleList([ResidualBlock2D(width, dropout) for _ in range(depth)])
        self.final_norm = nn.GroupNorm(_group_count(width), width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=attention_heads,
            dim_feedforward=width * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.axis_fusion = nn.TransformerEncoder(encoder_layer, num_layers=fusion_layers)
        self.resampler = LearnedQueryResampler(width, output_tokens, attention_heads, dropout)

    def forward(
        self,
        features: torch.Tensor,
        axis_mask: torch.Tensor,
        frequency_mask: torch.Tensor,
        cross_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 5 or features.shape[1] != 3:
            raise ValueError(
                f"wideband features must be [batch,3,channels,frequency,time], got {tuple(features.shape)}"
            )
        batch, axes, channels, frequency, time = features.shape
        if axis_mask.shape != (batch, 3):
            raise ValueError("wideband axis_mask shape mismatch")
        if frequency_mask.shape != (batch, frequency):
            raise ValueError("wideband frequency_mask shape mismatch")
        if cross_valid.shape != (batch, 3):
            raise ValueError("wideband cross_valid shape mismatch")
        valid_map = axis_mask[:, :, None, None, None] & frequency_mask[:, None, None, :, None]
        x = features * valid_map.to(features.dtype)
        x = x.reshape(batch * axes, channels, frequency, time)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = F.silu(self.final_norm(x))
        x = F.adaptive_avg_pool2d(x, (self.frequency_patches, self.time_patches))
        x = x.flatten(2).transpose(1, 2)
        patches_per_axis = self.frequency_patches * self.time_patches
        x = x.reshape(batch, axes, patches_per_axis, self.width)
        axis_ids = torch.arange(3, device=features.device)
        x = x + self.axis_embedding(axis_ids).view(1, 3, 1, self.width)
        # A learned validity embedding distinguishes synchronized cross spectra
        # from the intentionally zeroed unsynchronized channels.
        x = x + cross_valid.to(x.dtype).unsqueeze(-1).unsqueeze(-1) * 0.1
        source = x.reshape(batch, axes * patches_per_axis, self.width)
        axis_available = axis_mask & frequency_mask.any(dim=-1, keepdim=True)
        source_padding = (~axis_available).unsqueeze(-1).expand(-1, -1, patches_per_axis).reshape(batch, -1)
        safe_padding, valid_row = _safe_padding_mask(source_padding)
        source = self.axis_fusion(source, src_key_padding_mask=safe_padding)
        source = source * valid_row[:, None, None].to(source.dtype)
        tokens, token_mask = self.resampler(source, source_padding)
        return tokens, token_mask, source


class PhysicsFeatureEncoder(nn.Module):
    """Tokenize named scalar physics features and their explicit validity masks."""

    def __init__(
        self,
        *,
        feature_count: int,
        width: int,
        output_tokens: int,
        attention_heads: int,
        transformer_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.feature_count = int(feature_count)
        self.feature_embedding = nn.Parameter(torch.randn(feature_count, width) * (width**-0.5))
        self.scalar_projection = nn.Sequential(
            nn.Linear(2, width),
            nn.SiLU(),
            nn.Linear(width, width, bias=False),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=attention_heads,
            dim_feedforward=width * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.feature_fusion = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.resampler = LearnedQueryResampler(width, output_tokens, attention_heads, dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim != 2 or values.shape[-1] != self.feature_count or mask.shape != values.shape:
            raise ValueError("physics values/mask shape mismatch")
        valid = mask.to(values.dtype)
        scalar = torch.stack((values * valid, valid), dim=-1)
        source = self.scalar_projection(scalar) + self.feature_embedding.unsqueeze(0)
        source_padding = ~mask.to(torch.bool)
        safe_padding, valid_row = _safe_padding_mask(source_padding)
        source = self.feature_fusion(source, src_key_padding_mask=safe_padding)
        source = source * valid_row[:, None, None].to(source.dtype)
        tokens, token_mask = self.resampler(source, source_padding)
        return tokens, token_mask, source


@dataclass(slots=True)
class TokenizerOutput:
    tokens: torch.Tensor
    token_mask: torch.Tensor
    board_tokens: torch.Tensor
    modal_tokens: torch.Tensor
    wideband_tokens: torch.Tensor
    physics_tokens: torch.Tensor
    modal_latents: torch.Tensor
    wideband_latents: torch.Tensor
    physics_latents: torch.Tensor


class SixBoardVibrationTokenizer(nn.Module):
    """Reference-conditioned multi-resolution vibration tokenizer.

    It emits continuous tokens only. There is deliberately no class, anomaly,
    damage, severity, confidence, or threshold head anywhere in this module.
    """

    def __init__(
        self,
        *,
        width: int = 192,
        modal_depth: int = 6,
        modal_tokens_per_axis: int = 4,
        modal_tokens_per_board: int = 6,
        modal_kernel_sizes: list[int] | tuple[int, ...] = (7, 5, 5, 3, 3, 3),
        modal_dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        wideband_input_channels: int = 9,
        wideband_depth: int = 4,
        wideband_tokens_per_board: int = 6,
        wideband_frequency_patches: int = 4,
        wideband_time_patches: int = 4,
        physics_feature_count: int = 234,
        physics_tokens_per_board: int = 2,
        attention_heads: int = 4,
        branch_fusion_layers: int = 2,
        cross_board_layers: int = 2,
        dropout: float = 0.05,
        quality_features: int = 13,
        orientation_features: int = 11,
    ) -> None:
        super().__init__()
        if width % attention_heads != 0:
            raise ValueError("encoder width must be divisible by attention_heads")
        self.width = int(width)
        self.modal_tokens_per_board = int(modal_tokens_per_board)
        self.wideband_tokens_per_board = int(wideband_tokens_per_board)
        self.physics_tokens_per_board = int(physics_tokens_per_board)
        self.tokens_per_board = (
            self.modal_tokens_per_board + self.wideband_tokens_per_board + self.physics_tokens_per_board
        )
        self.modal_encoder = AxisAwareModalEncoder(
            width=width,
            depth=modal_depth,
            tokens_per_axis=modal_tokens_per_axis,
            output_tokens=modal_tokens_per_board,
            kernel_sizes=modal_kernel_sizes,
            dilations=modal_dilations,
            attention_heads=attention_heads,
            fusion_layers=1,
            dropout=dropout,
        )
        self.wideband_encoder = AxisAwareWidebandEncoder(
            input_channels=wideband_input_channels,
            width=width,
            depth=wideband_depth,
            output_tokens=wideband_tokens_per_board,
            attention_heads=attention_heads,
            fusion_layers=1,
            frequency_patches=wideband_frequency_patches,
            time_patches=wideband_time_patches,
            dropout=dropout,
        )
        self.physics_encoder = PhysicsFeatureEncoder(
            feature_count=physics_feature_count,
            width=width,
            output_tokens=physics_tokens_per_board,
            attention_heads=attention_heads,
            transformer_layers=1,
            dropout=dropout,
        )
        self.board_identity = nn.Embedding(6, width)
        self.board_role = nn.Embedding(2, width)
        self.token_position = nn.Embedding(self.tokens_per_board, width)
        self.branch_identity = nn.Embedding(3, width)
        self.axis_summary_embedding = nn.Embedding(3, width)
        self.sensor_position = nn.Sequential(
            nn.Linear(4, width),
            nn.SiLU(),
            nn.Linear(width, width, bias=False),
        )
        self.orientation_projection = nn.Sequential(
            nn.Linear(orientation_features, width),
            nn.SiLU(),
            nn.Linear(width, width, bias=False),
        )
        self.quality_projection = nn.Sequential(
            nn.Linear(quality_features, width),
            nn.SiLU(),
            nn.Linear(width, width, bias=False),
        )
        branch_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=attention_heads,
            dim_feedforward=width * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.branch_fusion = nn.TransformerEncoder(branch_layer, num_layers=branch_fusion_layers)
        self.reference_relation = nn.Sequential(
            nn.Linear(width * 4, width * 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width, bias=False),
        )
        cross_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=attention_heads,
            dim_feedforward=width * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_board_fusion = nn.TransformerEncoder(cross_layer, num_layers=cross_board_layers)
        self.output_norm = nn.LayerNorm(width)

    def forward(
        self,
        modal_values: torch.Tensor,
        modal_axis_mask: torch.Tensor,
        modal_sample_mask: torch.Tensor,
        wideband_features: torch.Tensor,
        wideband_frequency_mask: torch.Tensor,
        wideband_cross_valid: torch.Tensor,
        physics_features: torch.Tensor,
        physics_mask: torch.Tensor,
        sensor_positions: torch.Tensor,
        orientation_features: torch.Tensor,
        quality_features: torch.Tensor,
    ) -> TokenizerOutput:
        if modal_values.ndim != 4 or modal_values.shape[1:3] != (6, 3):
            raise ValueError(f"modal_values must be [batch,6,3,samples], got {tuple(modal_values.shape)}")
        batch, boards, _, modal_samples = modal_values.shape
        if boards != 6:
            raise ValueError("exactly six boards are required")
        flat_modal = modal_values.reshape(batch * boards, 3, modal_samples)
        flat_axis = modal_axis_mask.reshape(batch * boards, 3)
        flat_modal_mask = modal_sample_mask.reshape(batch * boards, modal_samples)
        modal_tokens, modal_mask, modal_latents = self.modal_encoder(flat_modal, flat_axis, flat_modal_mask)

        _, _, _, wideband_channels, frequency, time = wideband_features.shape
        flat_wideband = wideband_features.reshape(batch * boards, 3, wideband_channels, frequency, time)
        flat_frequency_mask = wideband_frequency_mask.reshape(batch * boards, frequency)
        flat_cross_valid = wideband_cross_valid.reshape(batch * boards, 3)
        wideband_tokens, wideband_mask, wideband_latents = self.wideband_encoder(
            flat_wideband,
            flat_axis,
            flat_frequency_mask,
            flat_cross_valid,
        )

        flat_physics = physics_features.reshape(batch * boards, -1)
        flat_physics_mask = physics_mask.reshape(batch * boards, -1)
        physics_tokens, physics_token_mask, physics_latents = self.physics_encoder(
            flat_physics,
            flat_physics_mask,
        )

        modal_tokens = modal_tokens.reshape(batch, boards, self.modal_tokens_per_board, self.width)
        wideband_tokens = wideband_tokens.reshape(batch, boards, self.wideband_tokens_per_board, self.width)
        physics_tokens = physics_tokens.reshape(batch, boards, self.physics_tokens_per_board, self.width)
        modal_mask = modal_mask.reshape(batch, boards, self.modal_tokens_per_board)
        wideband_mask = wideband_mask.reshape(batch, boards, self.wideband_tokens_per_board)
        physics_token_mask = physics_token_mask.reshape(batch, boards, self.physics_tokens_per_board)
        tokens = torch.cat((modal_tokens, wideband_tokens, physics_tokens), dim=2)
        token_mask = torch.cat((modal_mask, wideband_mask, physics_token_mask), dim=2)

        branch_ids = torch.cat(
            (
                torch.zeros(self.modal_tokens_per_board, dtype=torch.long, device=tokens.device),
                torch.ones(self.wideband_tokens_per_board, dtype=torch.long, device=tokens.device),
                torch.full(
                    (self.physics_tokens_per_board,), 2, dtype=torch.long, device=tokens.device
                ),
            )
        )
        board_ids = torch.arange(6, device=tokens.device)
        role_ids = torch.tensor([0, 1, 1, 1, 1, 1], device=tokens.device)
        token_ids = torch.arange(self.tokens_per_board, device=tokens.device)
        tokens = (
            tokens
            + self.branch_identity(branch_ids).view(1, 1, self.tokens_per_board, self.width)
            + self.board_identity(board_ids).view(1, 6, 1, self.width)
            + self.board_role(role_ids).view(1, 6, 1, self.width)
            + self.token_position(token_ids).view(1, 1, self.tokens_per_board, self.width)
        )

        position_valid = torch.isfinite(sensor_positions).all(dim=-1, keepdim=True)
        position_values = torch.nan_to_num(sensor_positions, nan=0.0)
        position_input = torch.cat((position_values, position_valid.to(position_values.dtype)), dim=-1)
        geometry = (
            self.sensor_position(position_input)
            + self.orientation_projection(orientation_features)
            + self.quality_projection(quality_features)
        )
        axis_summary = torch.zeros((batch, boards, self.width), device=tokens.device, dtype=tokens.dtype)
        for axis in range(3):
            axis_summary = axis_summary + modal_axis_mask[:, :, axis : axis + 1].to(tokens.dtype) * self.axis_summary_embedding.weight[axis]
        axis_count = modal_axis_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(tokens.dtype)
        axis_summary = axis_summary / axis_count
        tokens = tokens + (geometry + axis_summary).unsqueeze(2)

        flat_board_tokens = tokens.reshape(batch * boards, self.tokens_per_board, self.width)
        flat_board_padding = (~token_mask).reshape(batch * boards, self.tokens_per_board)
        safe_board_padding, board_valid = _safe_padding_mask(flat_board_padding)
        flat_board_tokens = self.branch_fusion(flat_board_tokens, src_key_padding_mask=safe_board_padding)
        flat_board_tokens = flat_board_tokens * board_valid[:, None, None].to(flat_board_tokens.dtype)
        board_tokens = flat_board_tokens.reshape(batch, boards, self.tokens_per_board, self.width)

        reference = board_tokens[:, 0:1].expand(-1, boards, -1, -1)
        relation_input = torch.cat(
            (board_tokens, reference, board_tokens - reference, board_tokens * reference),
            dim=-1,
        )
        board_tokens = board_tokens + self.reference_relation(relation_input)
        flat_tokens = board_tokens.reshape(batch, boards * self.tokens_per_board, self.width)
        flat_padding = (~token_mask).reshape(batch, boards * self.tokens_per_board)
        safe_padding, episode_valid = _safe_padding_mask(flat_padding)
        flat_tokens = self.cross_board_fusion(flat_tokens, src_key_padding_mask=safe_padding)
        flat_tokens = self.output_norm(flat_tokens)
        flat_tokens = flat_tokens * (~flat_padding).unsqueeze(-1).to(flat_tokens.dtype)
        flat_tokens = flat_tokens * episode_valid[:, None, None].to(flat_tokens.dtype)
        board_tokens = flat_tokens.reshape(batch, boards, self.tokens_per_board, self.width)

        return TokenizerOutput(
            tokens=flat_tokens,
            token_mask=(~flat_padding) & episode_valid[:, None],
            board_tokens=board_tokens,
            modal_tokens=modal_tokens,
            wideband_tokens=wideband_tokens,
            physics_tokens=physics_tokens,
            modal_latents=modal_latents.reshape(batch, boards, modal_latents.shape[1], self.width),
            wideband_latents=wideband_latents.reshape(batch, boards, wideband_latents.shape[1], self.width),
            physics_latents=physics_latents.reshape(batch, boards, physics_latents.shape[1], self.width),
        )


class MultiBranchReconstructionHead(nn.Module):
    """Temporary self-supervised heads used only during signal pretraining."""

    def __init__(
        self,
        *,
        width: int,
        modal_tokens: int,
        wideband_tokens: int,
        physics_tokens: int,
        modal_output_samples: int,
        spectral_output_frequency: int,
        spectral_output_time: int,
        physics_feature_count: int,
    ) -> None:
        super().__init__()
        self.modal_output_samples = int(modal_output_samples)
        self.spectral_output_frequency = int(spectral_output_frequency)
        self.spectral_output_time = int(spectral_output_time)
        self.physics_feature_count = int(physics_feature_count)
        self.modal = nn.Sequential(
            nn.Linear(width * modal_tokens, width * 2),
            nn.SiLU(),
            nn.Linear(width * 2, 3 * modal_output_samples),
        )
        self.wideband = nn.Sequential(
            nn.Linear(width * wideband_tokens, width * 2),
            nn.SiLU(),
            nn.Linear(width * 2, 3 * spectral_output_frequency * spectral_output_time),
        )
        self.physics = nn.Sequential(
            nn.Linear(width * physics_tokens, width * 2),
            nn.SiLU(),
            nn.Linear(width * 2, physics_feature_count),
        )

    def forward(
        self,
        output: TokenizerOutput,
        *,
        token_source: str = "branch",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, boards, _, width = output.board_tokens.shape
        if token_source == "branch":
            modal_tokens = output.modal_tokens
            wideband_tokens = output.wideband_tokens
            physics_tokens = output.physics_tokens
        elif token_source == "fused":
            modal_stop = output.modal_tokens.shape[2]
            wideband_stop = modal_stop + output.wideband_tokens.shape[2]
            modal_tokens = output.board_tokens[:, :, :modal_stop]
            wideband_tokens = output.board_tokens[:, :, modal_stop:wideband_stop]
            physics_tokens = output.board_tokens[:, :, wideband_stop:]
            if physics_tokens.shape[2] != output.physics_tokens.shape[2]:
                raise ValueError("fused token slices do not match branch token counts")
        else:
            raise ValueError("token_source must be 'branch' or 'fused'")
        modal = self.modal(modal_tokens.reshape(batch * boards, -1)).reshape(
            batch, boards, 3, self.modal_output_samples
        )
        wideband = self.wideband(wideband_tokens.reshape(batch * boards, -1)).reshape(
            batch,
            boards,
            3,
            self.spectral_output_frequency,
            self.spectral_output_time,
        )
        physics = self.physics(physics_tokens.reshape(batch * boards, -1)).reshape(
            batch, boards, self.physics_feature_count
        )
        return modal, wideband, physics
