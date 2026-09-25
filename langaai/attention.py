"""Raw and segment-sliced joint-stack attention.

There's one unrestricted attention matrix per layer over the concatenated
``[cls, ab, ag]`` sequence -- no block masking, no separate weight set for
the cross-chain block (see ``joint_attention.py``). Self/cross/epitope/
paratope views are therefore index-slices of that one matrix by segment,
not separately computed quantities.

Contents:
    attentions: Raw per-layer attention, one ``[n_heads, L, L]`` tensor per
        layer, per pair (``L = 1 + n_ab + n_ag`` for that pair -- no padding,
        see ``_batching._forward``).
    attention_block: A ``kind``-sliced view (``self_ab``, ``self_ag``,
        ``ab_to_ag`` i.e. paratope->epitope, ``ag_to_ab`` i.e.
        epitope->paratope, ``cls_to_ab``, ``cls_to_ag``), with `layer`/`head`
        selection or `average` reduction.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch

from ._batching import _forward
from .inputs import AbInput, AgEmbedding

__all__ = ["ATTENTION_BLOCK_KINDS", "attentions", "attention_block"]

ATTENTION_BLOCK_KINDS = (
    "self_ab", "self_ag", "ab_to_ag", "ag_to_ab", "cls_to_ab", "cls_to_ag",
)


@torch.no_grad()
def attentions(
    model: torch.nn.Module, device: torch.device,
    pairs: Sequence[Tuple[AbInput, AgEmbedding]],
) -> List[List[torch.Tensor]]:
    """Raw attention, every layer, for each pair.

    Returns ``result[i][layer]`` -> ``[n_heads, L, L]``, ``L = 1 + len(ab) +
    len(ag)`` for that pair.
    """
    results = []
    for ab, ag in pairs:
        out = _forward(model, device, ab, ag, return_dict=True)
        results.append([layer[0].cpu() for layer in out["attentions"]])
    return results


def _block_slices(kind: str, n_ab: int, n_ag: int) -> Tuple[slice, slice]:
    cls_s, ab_s, ag_s = slice(0, 1), slice(1, 1 + n_ab), slice(1 + n_ab, 1 + n_ab + n_ag)
    table = {
        "self_ab": (ab_s, ab_s), "self_ag": (ag_s, ag_s),
        "ab_to_ag": (ab_s, ag_s), "ag_to_ab": (ag_s, ab_s),
        "cls_to_ab": (cls_s, ab_s), "cls_to_ag": (cls_s, ag_s),
    }
    if kind not in table:
        raise ValueError(f"Unknown attention_block kind {kind!r}; expected one of {ATTENTION_BLOCK_KINDS}")
    return table[kind]


@torch.no_grad()
def attention_block(
    model: torch.nn.Module, device: torch.device,
    pairs: Sequence[Tuple[AbInput, AgEmbedding]],
    kind: str, layer: Optional[int] = None, head: Optional[int] = None,
    average: bool = False,
) -> Union[List[torch.Tensor], List[List[torch.Tensor]]]:
    """A segment-sliced view of the raw attention matrix for each pair.

    ``layer``: a specific layer index, or ``None`` for every layer.
    ``head``: a specific head index, or ``None`` for every head.
    ``average``: average over whichever of layer/head was left as ``None``
        (both, if neither is given).

    Returns one ``[rows, cols]`` tensor per pair if ``layer`` is given or
    ``average=True``; otherwise one list of per-layer ``[rows, cols]`` (or
    ``[n_heads, rows, cols]`` if ``head`` is also unspecified and
    ``average=False``) tensors per pair.
    """
    results = []
    for ab, ag in pairs:
        out = _forward(model, device, ab, ag, return_dict=True)
        rows, cols = _block_slices(kind, len(ab), len(ag))
        layers = out["attentions"] if layer is None else [out["attentions"][layer]]

        blocks = []
        for layer_attn in layers:
            block = layer_attn[0][:, rows, :][:, :, cols]  # [n_heads, r, c]
            if head is not None:
                block = block[head]
            elif average:
                block = block.mean(dim=0)
            blocks.append(block.cpu())

        if layer is not None:
            results.append(blocks[0])
        elif average:
            results.append(torch.stack(blocks).mean(dim=0))
        else:
            results.append(blocks)
    return results
