"""The five embedding flavors ``JointCDRMLM`` can produce, plus a mean-pool
helper for turning a per-residue embedding into one sequence-level vector.

Contents:
    antibody_embedding_blind: Raw frozen ESM-2, antigen-blind -- what
        ``model._encode_antibody`` computes, thrown away post-adapter inside
        ``forward()`` today. A legitimate, distinct baseline embedding, kept
        here deliberately rather than left inaccessible.
    antibody_embedding_conditioned: Post-joint-stack ``ab`` -- antigen-aware.
    antigen_embedding_blind: The raw ``ag_emb`` input, antibody-blind
        (trivial passthrough of :attr:`AgEmbedding.embedding`, included for
        API symmetry with the other three).
    antigen_embedding_conditioned: Post-joint-stack ``ag`` -- "paratope-aware".
    cls_embedding: The one native pair-level pooled vector (``out["cls"]``).
        This is the representation to build a downstream predictor on -- see
        MODEL_CARD.md for what has and hasn't been validated on top of it.
    mean_pool: ``[L, dim] -> [dim]`` mean over all positions -- every tensor
        this module returns is already exactly the item's own length (see
        ``_batching._forward``'s docstring), so no pad mask is needed here.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from ._batching import _forward
from .inputs import AbInput, AgEmbedding

__all__ = [
    "antibody_embedding_blind", "antibody_embedding_conditioned",
    "antigen_embedding_blind", "antigen_embedding_conditioned",
    "cls_embedding", "mean_pool",
]


@torch.no_grad()
def antibody_embedding_blind(
    model: torch.nn.Module, device: torch.device, ab_list: Sequence[AbInput],
) -> List[torch.Tensor]:
    """Plain frozen ESM-2 over the antibody alone, no antigen context."""
    results = []
    for ab in ab_list:
        tokens = torch.from_numpy(ab.tokens).long().unsqueeze(0).to(device)
        pad = torch.zeros(1, len(ab), dtype=torch.bool, device=device)
        hidden = model._encode_antibody(tokens, pad)[0]
        results.append(hidden.cpu())
    return results


@torch.no_grad()
def antibody_embedding_conditioned(
    model: torch.nn.Module, device: torch.device,
    pairs: Sequence[Tuple[AbInput, AgEmbedding]],
) -> List[torch.Tensor]:
    """Post-joint-stack antibody embedding -- antigen-conditioned."""
    return [_forward(model, device, ab, ag, return_dict=True)["ab"][0].cpu() for ab, ag in pairs]


def antigen_embedding_blind(ag_list: Sequence[AgEmbedding]) -> List[torch.Tensor]:
    """The raw per-residue antigen embedding, antibody-blind -- exactly
    what was passed in, unchanged."""
    return [ag.embedding for ag in ag_list]


@torch.no_grad()
def antigen_embedding_conditioned(
    model: torch.nn.Module, device: torch.device,
    pairs: Sequence[Tuple[AbInput, AgEmbedding]],
) -> List[torch.Tensor]:
    """Post-joint-stack antigen embedding -- "paratope-aware"."""
    return [_forward(model, device, ab, ag, return_dict=True)["ag"][0].cpu() for ab, ag in pairs]


@torch.no_grad()
def cls_embedding(
    model: torch.nn.Module, device: torch.device,
    pairs: Sequence[Tuple[AbInput, AgEmbedding]],
) -> List[torch.Tensor]:
    """The native pair-level pooled vector."""
    return [_forward(model, device, ab, ag, return_dict=True)["cls"][0, 0].cpu() for ab, ag in pairs]


def mean_pool(embedding: torch.Tensor) -> torch.Tensor:
    """``[L, dim] -> [dim]``, mean over all positions."""
    return embedding.mean(dim=0)
