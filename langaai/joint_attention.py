"""Unrestricted pre-LN self-attention over the concatenated pair.

Every residue attends to every other residue, within its own chain and across to
the partner: there is no block masking and no separate weight set for the
cross-chain block, so antibody-antigen attention and within-chain attention are
slices of one matrix rather than separate quantities. Pad positions are zeroed
after each layer so downstream pooling stays clean.

Contents:
    StackConfig: Hyperparameter dataclass (dim, n_layers, n_heads, dropout, ffn_mult).
    SwiGLU: Gated feed-forward block.
        forward: Gate/value split, silu-gate, project back down.
    JointLayer: One pre-LN self-attention + SwiGLU block.
        forward: Attend over the full joint sequence, then FFN, both residual.
    JointStack: N stacked JointLayers + final LayerNorm.
        forward: Runs the stack, zeroing pad positions after every layer;
            optionally returns each layer's attention weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["JointStack", "StackConfig"]


@dataclass(frozen=True)
class StackConfig:
    """Joint stack hyperparameters.

    Attributes:
        dim: Model width. Pinned to the encoder width so the LM head can tie.
        n_layers: Number of self-attention layers.
        n_heads: Attention heads per layer.
        dropout: Applied to attention and FFN output.
        ffn_mult: Hidden multiplier for the SwiGLU feed-forward block.
    """

    dim: int
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    ffn_mult: int = 4


class SwiGLU(nn.Module):
    """Gated feed-forward block."""

    def __init__(self, dim: int, mult: int, dropout: float) -> None:
        super().__init__()
        hidden = dim * mult
        self.w_in = nn.Linear(dim, hidden * 2)
        self.w_out = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.w_in(x).chunk(2, dim=-1)
        return self.drop(self.w_out(F.silu(gate) * value))


class JointLayer(nn.Module):
    """One pre-LN self-attention layer."""

    def __init__(self, cfg: StackConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(cfg.dim)
        self.attn = nn.MultiheadAttention(
            cfg.dim, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(cfg.dim)
        self.ffn = SwiGLU(cfg.dim, cfg.ffn_mult, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.attn_norm(x)
        attended, weights = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x + self.drop(attended)
        return x + self.ffn(self.ffn_norm(x)), weights


class JointStack(nn.Module):
    """N unrestricted self-attention layers over the joint sequence."""

    def __init__(self, cfg: StackConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([JointLayer(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        need_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the stack.

        Args:
            x: ``[B, L, dim]`` joint sequence.
            key_padding_mask: ``[B, L]`` bool, ``True`` at pad positions.
            need_weights: if True, also return each layer's attention
                weights, ``[B, n_heads, L, L]``.

        Returns:
            ``[B, L, dim]`` with pad positions zeroed. If ``need_weights``,
            a ``(hidden, attentions)`` tuple, one attention tensor per layer.
        """
        keep = (~key_padding_mask).unsqueeze(-1).to(x.dtype)
        x = x * keep
        attentions: list[torch.Tensor] = []
        for layer in self.layers:
            x, weights = layer(x, key_padding_mask, need_weights=need_weights)
            x = x * keep
            if need_weights:
                attentions.append(weights)
        x = self.final_norm(x) * keep
        if need_weights:
            return x, attentions
        return x
