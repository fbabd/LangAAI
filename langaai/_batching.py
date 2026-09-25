"""The single-pair forward primitive every public function goes through.

``_forward`` deliberately runs one (antibody, antigen) pair per model call
rather than padding a list of differently-sized pairs into one tensor batch.
The joint sequence's pad positions land in the *middle* of the layout --
``cls`` + antibody block + antigen block, each padded to its own batch
maximum -- so a padded multi-item batch needs careful boolean-mask indexing
to recover each item's un-padded attention and embedding slices. A per-pair
forward pass sidesteps that whole class of bug: every tensor this package
returns is exactly the requested item's own length, always, so no caller
ever has to strip padding.

Every public function still accepts a *list* of pairs, so call sites are
batch-shaped; the list is just looped over internally. If that Python loop
becomes your bottleneck across many thousands of pairs, sort your pairs by
joint length so same-length pairs sit together, and drive ``model.forward``
directly with your own padded batches.

Contents:
    _forward: One (antibody, antigen) pair through the model, no padding.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from .inputs import SEG_AG, SEG_CLS, AbInput, AgEmbedding, MaskedAbInput

__all__: list[str] = []


def _forward(
    model: torch.nn.Module,
    device: torch.device,
    ab: Union[AbInput, MaskedAbInput],
    ag: AgEmbedding,
    return_dict: bool = True,
):
    """Run one (antibody, antigen) pair through ``model.forward``, batch
    size 1, no padding. ``ab`` may carry masked positions (:class:`MaskedAbInput`)
    or not (:class:`AbInput`, used as-is for embeddings/attention that don't
    need masking)."""
    base = ab.base if isinstance(ab, MaskedAbInput) else ab
    tokens = base.tokens.copy()
    if isinstance(ab, MaskedAbInput) and ab.positions:
        tokens[list(ab.positions)] = model.mask_token_id

    n_ab, n_ag = len(tokens), len(ag)
    ab_tokens = torch.from_numpy(tokens).long().unsqueeze(0).to(device)
    ab_pad = torch.zeros(1, n_ab, dtype=torch.bool, device=device)
    ag_emb = ag.embedding.unsqueeze(0).to(device)
    ag_pad = torch.zeros(1, n_ag, dtype=torch.bool, device=device)

    segment = np.concatenate(
        [[SEG_CLS], base.segment, np.full(n_ag, SEG_AG, dtype=np.int8)]
    ).astype(np.int64)
    segment_ids = torch.from_numpy(segment).unsqueeze(0).to(device)

    with torch.no_grad():
        return model(
            ab_tokens=ab_tokens, ab_pad=ab_pad, ag_emb=ag_emb, ag_pad=ag_pad,
            segment_ids=segment_ids, return_dict=return_dict,
        )
