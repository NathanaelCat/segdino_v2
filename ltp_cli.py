"""Lightweight token-pyramid and cross-linear interaction modules.

This file contains the new, deliberately small spatial branch for the
OSD-only ``DINO-LTP-CLI-001`` candidate.  It does not import the decoder so
that it can be used by ``dpt.py`` without creating an import cycle.

The implementation uses kernelized linear attention throughout.  No
``[N_query, N_key]`` pairwise attention matrix is materialized.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


KERNEL_EPS = 1e-6


def _check_sequence(value: torch.Tensor, name: str) -> None:
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [B,N,C], got {tuple(value.shape)}")


def _split_heads(value: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Convert [B,N,D] into [B,H,N,D/H]."""
    _check_sequence(value, "attention tensor")
    batch, tokens, channels = value.shape
    if channels % num_heads != 0:
        raise ValueError(
            f"attention dimension {channels} is not divisible by num_heads={num_heads}"
        )
    return value.reshape(batch, tokens, num_heads, channels // num_heads).transpose(1, 2)


def _merge_heads(value: torch.Tensor) -> torch.Tensor:
    """Convert [B,H,N,D] into [B,N,H*D]."""
    if value.ndim != 4:
        raise ValueError(f"headed attention tensor must be 4D, got {tuple(value.shape)}")
    batch, heads, tokens, head_dim = value.shape
    return value.transpose(1, 2).reshape(batch, tokens, heads * head_dim)


def _relu_kernel(value: torch.Tensor) -> torch.Tensor:
    return F.relu(value) + KERNEL_EPS


def _linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Compute ReLU-kernel attention without constructing pairwise scores.

    Args:
        query: [B,H,Nq,D]
        key: [B,H,Nk,D]
        value: [B,H,Nk,D]
    Returns:
        [B,H,Nq,D]
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("linear attention expects [B,H,N,D] tensors")
    if key.shape != value.shape:
        raise ValueError(
            f"key/value shape mismatch: {tuple(key.shape)} vs {tuple(value.shape)}"
        )
    if query.shape[0] != key.shape[0] or query.shape[1] != key.shape[1]:
        raise ValueError("query and key batch/head dimensions must match")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query/key head dimensions must match")

    # These are [B,H,D,D] and [B,H,D], respectively.  They replace the
    # conventional [B,H,Nq,Nk] attention score tensor.
    key_value = torch.einsum("bhnd,bhne->bhde", key, value)
    key_sum = key.sum(dim=2)
    denominator = torch.einsum("bhnd,bhd->bhn", query, key_sum).unsqueeze(-1)
    return torch.einsum("bhnd,bhde->bhne", query, key_value) / (denominator + KERNEL_EPS)


class LiteLinearAttentionBlock(nn.Module):
    """Pre-LN ReLU-kernel multi-head linear self-attention plus FFN."""

    def __init__(self, channels: int, num_heads: int = 4, ffn_ratio: int = 2):
        super().__init__()
        channels = int(channels)
        num_heads = int(num_heads)
        hidden = int(channels * ffn_ratio)
        if channels <= 0 or num_heads <= 0 or channels % num_heads != 0:
            raise ValueError(
                f"invalid attention dimensions channels={channels}, heads={num_heads}"
            )
        if hidden <= 0:
            raise ValueError(f"ffn_ratio must produce a positive hidden size, got {ffn_ratio}")

        self.channels = channels
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"LiteLinearAttentionBlock expects [B,C,H,W], got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(
                f"LiteLinearAttentionBlock expected C={self.channels}, got C={channels}"
            )

        tokens = x.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens)
        query = _split_heads(_relu_kernel(self.q_proj(normalized)), self.num_heads)
        key = _split_heads(_relu_kernel(self.k_proj(normalized)), self.num_heads)
        value = _split_heads(self.v_proj(normalized), self.num_heads)
        attended = _merge_heads(_linear_attention(query, key, value))
        tokens = tokens + self.out_proj(attended)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class PatchMerge(nn.Module):
    """2x2 token merge using LN and Linear, without spatial convolutions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.norm = nn.LayerNorm(4 * self.in_channels)
        self.reduction = nn.Linear(4 * self.in_channels, self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"PatchMerge expects [B,C,H,W], got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"PatchMerge expected C={self.in_channels}, got C={channels}"
            )
        if height % 2 or width % 2:
            raise ValueError(f"PatchMerge requires even H/W, got {(height, width)}")

        # [B,H/2,W/2,2,2,C] -> [B,H/2,W/2,4C]
        tokens = x.reshape(batch, channels, height // 2, 2, width // 2, 2)
        tokens = tokens.permute(0, 2, 4, 3, 5, 1).reshape(
            batch, height // 2, width // 2, 4 * channels
        )
        tokens = self.reduction(self.norm(tokens))
        return tokens.permute(0, 3, 1, 2).contiguous()


def fixed_2d_sincos(height: int, width: int, channels: int, device, dtype) -> torch.Tensor:
    """Build fixed 2D sine/cosine positions in a normalized common grid.

    Cell centers are normalized to [0, 1] independently at each scale.  Thus
    corresponding locations in S4/S8/S16 use the same coordinate system.
    The returned tensor is [height*width, channels] and has no parameters.
    """
    height, width, channels = int(height), int(width), int(channels)
    if height <= 0 or width <= 0 or channels <= 0 or channels % 4:
        raise ValueError(
            f"fixed_2d_sincos requires positive H/W and channels divisible by 4; "
            f"got {(height, width, channels)}"
        )

    def encode(position: torch.Tensor, dimension: int) -> torch.Tensor:
        half = dimension // 2
        frequency = torch.arange(half, device=device, dtype=torch.float32)
        frequency = torch.pow(
            torch.tensor(10000.0, device=device, dtype=torch.float32),
            -frequency / max(1, half),
        )
        phase = (2.0 * math.pi) * position.float().unsqueeze(1) * frequency.unsqueeze(0)
        return torch.cat((phase.sin(), phase.cos()), dim=1)

    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    y_encoding = encode(y, channels // 2)
    x_encoding = encode(x, channels // 2)
    y_encoding = y_encoding[:, None, :].expand(height, width, -1)
    x_encoding = x_encoding[None, :, :].expand(height, width, -1)
    return torch.cat((y_encoding, x_encoding), dim=-1).reshape(height * width, channels).to(dtype)


class LightweightTokenPyramid(nn.Module):
    """S4/S8/S16 RGB token pyramid used as shared CLI memory."""

    def __init__(
        self,
        s4_channels: int = 64,
        s8_channels: int = 96,
        s16_channels: int = 128,
        memory_channels: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.s4_channels = int(s4_channels)
        self.s8_channels = int(s8_channels)
        self.s16_channels = int(s16_channels)
        self.memory_channels = int(memory_channels)
        self.patch_embed = nn.Conv2d(3, self.s4_channels, kernel_size=4, stride=4, bias=False)
        self.s4_block = LiteLinearAttentionBlock(self.s4_channels, num_heads=num_heads)
        self.merge_s8 = PatchMerge(self.s4_channels, self.s8_channels)
        self.s8_block = LiteLinearAttentionBlock(self.s8_channels, num_heads=num_heads)
        self.merge_s16 = PatchMerge(self.s8_channels, self.s16_channels)
        self.s16_block = LiteLinearAttentionBlock(self.s16_channels, num_heads=num_heads)
        self.memory_projections = nn.ModuleList(
            [
                nn.Linear(self.s4_channels, self.memory_channels),
                nn.Linear(self.s8_channels, self.memory_channels),
                nn.Linear(self.s16_channels, self.memory_channels),
            ]
        )
        # Zero initialization leaves the fixed positional encoding as the
        # initial scale identity while retaining three learned embeddings.
        self.scale_embeddings = nn.Parameter(torch.zeros(3, self.memory_channels))

    def forward(self, image: torch.Tensor, return_trace: bool = False):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"LTP expects RGB [B,3,H,W], got {tuple(image.shape)}")
        if image.shape[-2] % 16 or image.shape[-1] % 16:
            raise ValueError(
                "LTP input H/W must be divisible by 16 so S4/S8/S16 align with DINO S16"
            )

        s4 = self.s4_block(self.patch_embed(image))
        s8 = self.s8_block(self.merge_s8(s4))
        s16 = self.s16_block(self.merge_s16(s8))
        maps = (s4, s8, s16)

        memory_parts = []
        position_parts = []
        for index, (feature, projection) in enumerate(zip(maps, self.memory_projections)):
            batch, channels, height, width = feature.shape
            tokens = feature.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
            tokens = projection(tokens)
            positions = fixed_2d_sincos(
                height,
                width,
                self.memory_channels,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            memory_parts.append(tokens)
            scale_embedding = self.scale_embeddings[index].to(dtype=tokens.dtype).view(1, 1, -1)
            position_parts.append(
                (positions.unsqueeze(0) + scale_embedding).expand(batch, -1, -1)
            )
        # Keep content and geometry separate.  CLI applies the geometry only
        # after Wk to K; V remains a pure content projection.
        memory = torch.cat(memory_parts, dim=1)
        memory_position = torch.cat(position_parts, dim=1)

        if not return_trace:
            return memory
        return memory, {
            "s4": s4,
            "s8": s8,
            "s16": s16,
            "memory": memory,
            "memory_position": memory_position,
        }


class CrossLinearInteraction(nn.Module):
    """Let each DINO level read the shared LTP memory with linear attention."""

    materializes_pairwise_attention = False

    def __init__(
        self,
        query_channels: int = 256,
        memory_channels: int = 64,
        interaction_channels: int = 64,
        num_heads: int = 4,
        num_queries: int = 4,
    ):
        super().__init__()
        if interaction_channels % num_heads:
            raise ValueError(
                f"interaction_channels={interaction_channels} must be divisible by "
                f"num_heads={num_heads}"
            )
        self.query_channels = int(query_channels)
        self.memory_channels = int(memory_channels)
        self.interaction_channels = int(interaction_channels)
        self.num_heads = int(num_heads)
        self.query_projections = nn.ModuleList(
            [nn.Linear(query_channels, interaction_channels) for _ in range(num_queries)]
        )
        self.output_projections = nn.ModuleList(
            [nn.Linear(interaction_channels, query_channels) for _ in range(num_queries)]
        )
        self.key_projection = nn.Linear(memory_channels, interaction_channels)
        self.value_projection = nn.Linear(memory_channels, interaction_channels)
        self.gammas = nn.ParameterList(
            [nn.Parameter(torch.zeros(())) for _ in range(num_queries)]
        )

    def forward(
        self,
        projected_features: Sequence[torch.Tensor],
        memory: torch.Tensor,
        memory_position: torch.Tensor,
        query_position: torch.Tensor,
        return_trace: bool = False,
    ):
        if len(projected_features) != len(self.query_projections):
            raise ValueError(
                f"CLI expects {len(self.query_projections)} DINO feature maps, "
                f"got {len(projected_features)}"
            )
        _check_sequence(memory, "CLI memory")
        _check_sequence(memory_position, "CLI memory position")
        if memory_position.shape != memory.shape:
            raise ValueError(
                "CLI memory content/position shapes must match: "
                f"{tuple(memory.shape)} vs {tuple(memory_position.shape)}"
            )
        _check_sequence(query_position, "CLI query position")
        if query_position.shape[0] not in (1, memory.shape[0]):
            raise ValueError(
                "CLI query position batch must be 1 or match memory batch, "
                f"got {query_position.shape[0]} vs {memory.shape[0]}"
            )
        if query_position.shape[-1] != self.interaction_channels:
            raise ValueError(
                "CLI query position width must equal interaction width, "
                f"got {query_position.shape[-1]} vs {self.interaction_channels}"
            )

        # Geometry enters K after Wk and enters Q after Wq.  V is deliberately
        # content-only, so position cannot leak into the returned detail value.
        key = _split_heads(
            _relu_kernel(self.key_projection(memory) + memory_position), self.num_heads
        )
        value = _split_heads(self.value_projection(memory), self.num_heads)

        calibrated = []
        interaction_outputs = []
        for feature, query_projection, output_projection, gamma in zip(
            projected_features,
            self.query_projections,
            self.output_projections,
            self.gammas,
        ):
            if feature.ndim != 4:
                raise ValueError(
                    f"CLI DINO features must be [B,C,H,W], got {tuple(feature.shape)}"
                )
            batch, channels, height, width = feature.shape
            if channels != self.query_channels:
                raise ValueError(
                    f"CLI expected DINO C={self.query_channels}, got C={channels}"
                )
            tokens = feature.flatten(2).transpose(1, 2)
            if query_position.shape[1] != tokens.shape[1]:
                raise ValueError(
                    "CLI query position token count must match each DINO grid, "
                    f"got {query_position.shape[1]} vs {tokens.shape[1]}"
                )
            query = _split_heads(
                _relu_kernel(query_projection(tokens) + query_position), self.num_heads
            )
            output = _merge_heads(_linear_attention(query, key, value))
            delta = output_projection(output)
            calibrated_tokens = tokens + gamma * delta
            calibrated_map = calibrated_tokens.transpose(1, 2).reshape(
                batch, channels, height, width
            )
            calibrated.append(calibrated_map)
            interaction_outputs.append(delta)

        if not return_trace:
            return tuple(calibrated)
        return tuple(calibrated), {
            "memory": memory,
            "memory_position": memory_position,
            "query_position": query_position,
            "calibrated": tuple(calibrated),
            "interaction_outputs": tuple(interaction_outputs),
        }


__all__ = [
    "CrossLinearInteraction",
    "LightweightTokenPyramid",
    "LiteLinearAttentionBlock",
    "PatchMerge",
    "fixed_2d_sincos",
]
