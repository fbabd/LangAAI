"""Masked-residue prediction and sequence scoring.

Contents:
    PositionPrediction: One masked position's top-k amino acids.
    predict_masked: Top-k residues at each masked position, per pair.
    residue_probabilities: Full amino-acid distribution at every masked
        position, per pair.
    score_sequence: Pseudo-log-likelihood of the (masked) true residues
        given the antigen -- explicitly NOT a validated affinity predictor,
        see the module-level disclaimer below.

**score_sequence disclaimer**: this is masked-reconstruction likelihood --
"how well does the joint stack predict these residues from this antigen +
the rest of the sequence" -- not a validated binding-affinity score. No
evaluation behind this model established that connection, so a higher score
means "more consistent with the model's learned CDR-given-antigen
distribution", not "binds better".

If you want affinity, fit your own regressor on ``cls_embedding`` against
real affinity labels. A fitted, supervised model is a different thing from a
reconstruction likelihood, which is all ``score_sequence`` reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from ._batching import _forward
from .inputs import AbInput, AgEmbedding, MaskedAbInput

__all__ = ["PositionPrediction", "predict_masked", "residue_probabilities", "score_sequence"]

#: The 20 canonical amino acids plus X ("unresolved/ambiguous residue").
#: Probabilities are a softmax over *these* logits only, not over ESM-2's
#: full vocabulary, which also contains special and non-standard tokens that
#: are not valid residues. Drop ``X`` yourself if you are designing a
#: sequence rather than reconstructing one.
_CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWYX"


def _amino_acid_ids(tokenizer: PreTrainedTokenizerBase) -> Tuple[List[int], List[str]]:
    vocab = tokenizer.get_vocab()
    letters = [c for c in _CANONICAL_AA if c in vocab]
    return [vocab[c] for c in letters], letters


@dataclass(frozen=True)
class PositionPrediction:
    position: int
    top: List[Tuple[str, float]]  # (letter, probability), sorted descending


def predict_masked(
    model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, device: torch.device,
    pairs: Sequence[Tuple[MaskedAbInput, AgEmbedding]], top_k: int = 5,
) -> List[List[PositionPrediction]]:
    """Top-``top_k`` residues at each masked position, for each pair.

    ``pairs`` is a list of ``(masked_ab_input, ag_embedding)`` -- pass a
    single-item list for one pair. Each ``MaskedAbInput`` normally comes
    from :meth:`AbInput.mask_positions` / :meth:`AbInput.mask_region`.
    """
    aa_ids, aa_letters = _amino_acid_ids(tokenizer)
    aa_ids_t = torch.tensor(aa_ids)

    results: List[List[PositionPrediction]] = []
    for ab, ag in pairs:
        out = _forward(model, device, ab, ag, return_dict=True)
        logits = out["logits"][0]  # [n_ab, vocab]
        per_pair = []
        for pos in ab.positions:
            probs = torch.softmax(logits[pos, aa_ids_t], dim=-1)
            order = torch.argsort(probs, descending=True)[:top_k]
            top = [(aa_letters[i], float(probs[i])) for i in order.tolist()]
            per_pair.append(PositionPrediction(position=pos, top=top))
        results.append(per_pair)
    return results


def residue_probabilities(
    model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, device: torch.device,
    pairs: Sequence[Tuple[MaskedAbInput, AgEmbedding]],
) -> List[Dict[int, Dict[str, float]]]:
    """Full amino-acid distribution (all 20+X, not just top-k) at every
    masked position, for each pair -- ``result[i][position]`` -> ``{letter:
    prob}``."""
    aa_ids, aa_letters = _amino_acid_ids(tokenizer)
    aa_ids_t = torch.tensor(aa_ids)

    results: List[Dict[int, Dict[str, float]]] = []
    for ab, ag in pairs:
        out = _forward(model, device, ab, ag, return_dict=True)
        logits = out["logits"][0]
        per_pair = {}
        for pos in ab.positions:
            probs = torch.softmax(logits[pos, aa_ids_t], dim=-1)
            per_pair[pos] = {letter: float(p) for letter, p in zip(aa_letters, probs.tolist())}
        results.append(per_pair)
    return results


def score_sequence(
    model: torch.nn.Module, device: torch.device,
    pairs: Sequence[Tuple[AbInput, AgEmbedding]],
    positions: Optional[Sequence[Optional[Sequence[int]]]] = None,
) -> List[float]:
    """Mean log-likelihood of the true residues at the scored positions,
    given the antigen -- **not a validated affinity predictor**, see the
    module docstring.

    All scored positions are masked *simultaneously* in one forward pass per
    pair -- not per-position leave-one-out pseudo-likelihood. This matters
    when comparing candidate sequences: with a whole span masked, the
    forward pass does not see any of the candidate's residues in that span,
    so every candidate for the same span scores identically. To let a
    candidate's residues condition on each other, call this once per
    position with only that position masked and sum the results.

    Args:
        model: The loaded :class:`~langaai.model.JointCDRMLM`.
        device: Where to run the forward passes.
        pairs: ``(ab_input, ag_embedding)`` pairs.
        positions: Per pair, the positions to score, or ``None`` for every
            position in the antibody. Pass a CDR3 span's indices to score
            just that region.

    Returns:
        One mean log-likelihood per pair, in nats. Higher is better.
    """
    if positions is None:
        positions = [None] * len(pairs)

    scores: List[float] = []
    for (ab, ag), pos in zip(pairs, positions):
        pos = list(range(len(ab))) if pos is None else list(pos)
        masked = ab.mask_positions(pos)
        out = _forward(model, device, masked, ag, return_dict=True)
        logits = out["logits"][0][pos]  # [n_pos, vocab]
        true_ids = torch.from_numpy(ab.tokens[pos]).long()
        log_probs = torch.log_softmax(logits, dim=-1)
        token_logp = log_probs[torch.arange(len(pos)), true_ids]
        scores.append(float(token_logp.mean().item()))
    return scores
