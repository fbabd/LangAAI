"""Masked-LM head whose output projection is tied to the encoder embeddings.

The tie is load-bearing rather than an economy. The joint stack is randomly
initialised and the supervised signal is thin — roughly 1.7M masked tokens per
epoch. An untied head would have to learn amino-acid output geometry at the same
time as learning the conditioning. Tying hands it that geometry at step 0, so
the stack only has to learn what the antigen implies.

This is also why ``d`` is pinned to the encoder width: the tie requires it.

Contents:
    TiedLMHead: Dense+GELU+LayerNorm transform, then a Linear whose weight is
        tied to the (frozen) input token embedding.
        __init__: Validates dim == embedding width, ties decoder.weight.
        forward: Transform then decode to vocab logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["TiedLMHead"]


class TiedLMHead(nn.Module):
    """Dense transform, then a projection sharing the embedding matrix."""

    def __init__(self, dim: int, embedding: nn.Embedding) -> None:
        super().__init__()
        if embedding.embedding_dim != dim:
            raise ValueError(
                f"Tied head needs dim == embedding width, got {dim} vs "
                f"{embedding.embedding_dim}. `d` is pinned to the encoder width."
            )
        self.dense = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.decoder = nn.Linear(dim, embedding.num_embeddings, bias=False)
        self.decoder.weight = embedding.weight  # tied, and frozen with the tower
        self.bias = nn.Parameter(torch.zeros(embedding.num_embeddings))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(nn.functional.gelu(self.dense(x)))
        return self.decoder(h) + self.bias
