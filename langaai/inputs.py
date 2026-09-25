"""Antibody/antigen input prep: tokenization, masking, and live antigen
embedding.

Contents:
    SEG_CLS/SEG_HEAVY/SEG_LIGHT/SEG_AG: Segment ids (0/1/2/3). The model's
        `segment_ids` input has no meaning outside this exact numbering --
        it indexes a learned embedding table, so renumbering silently
        changes the model's inputs.
    AbInput: One antibody's tokenized heavy(+light) chain, unmasked.
    MaskedAbInput: An AbInput with some positions replaced by <mask>.
    AgEmbedding: One antigen's per-residue ESM-2 embedding.
    encode_antibody_sequence: heavy(+light) string(s) -> AbInput.
    embed_antigen: A live single-sequence ESM-2 pass for a new antigen.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import EsmModel, PreTrainedTokenizerBase

__all__ = [
    "SEG_CLS", "SEG_HEAVY", "SEG_LIGHT", "SEG_AG", "TRAINED_MAX_JOINT_LEN",
    "AbInput", "MaskedAbInput", "AgEmbedding",
    "encode_antibody_sequence", "embed_antigen",
]

SEG_CLS, SEG_HEAVY, SEG_LIGHT, SEG_AG = 0, 1, 2, 3

#: Joint-sequence length (cls + antibody + antigen) the model was trained
#: at. Longer inputs run but are out of distribution.
TRAINED_MAX_JOINT_LEN = 1024

#: Rough heavy+light variable-domain length, used only to decide when an
#: antigen is long enough to warn about.
_TYPICAL_ANTIBODY_LEN = 240


@dataclass(frozen=True)
class AbInput:
    """One antibody's tokenized heavy(+light) chain, no masking applied.

    ``tokens``/``segment`` are indexed ``0..len(tokens)-1`` over the heavy
    chain followed by the light chain (if present) -- the coordinate space
    ``mask_positions``/``mask_region`` and every function in ``predict.py``/
    ``embeddings.py``/``attention.py`` use.
    """

    heavy: str
    light: Optional[str]
    tokens: np.ndarray   # int64, no CLS/EOS -- the tower adds/strips them itself
    segment: np.ndarray  # int8, SEG_HEAVY/SEG_LIGHT per position

    def __len__(self) -> int:
        return len(self.tokens)

    def mask_positions(self, positions: Sequence[int]) -> "MaskedAbInput":
        """Replace the given positions with ``<mask>``, tracked for scoring."""
        positions = tuple(sorted({int(p) for p in positions}))
        if positions and (positions[0] < 0 or positions[-1] >= len(self)):
            raise ValueError(f"mask position out of range for a {len(self)}-token input")
        return MaskedAbInput(self, positions)

    def mask_region(self, name: str, spans: Sequence[Tuple[int, int]]) -> "MaskedAbInput":
        """Mask one or more half-open ``(start, end)`` spans.

        Spans are always caller-supplied. This package bundles no
        CDR-numbering tool, so it cannot work out where CDR3 starts from a
        raw sequence: get spans from your own numbering step (ANARCI,
        IMGT/V-QUEST, or whatever annotation your data already carries).
        ``name`` is a caller-facing label carried onto the result for
        readability; it isn't interpreted.

        Args:
            name: A label for the region, e.g. ``"cdr3"``. Not interpreted.
            spans: Half-open ``(start, end)`` index pairs into this input's
                own coordinate space -- heavy chain followed by light.
        """
        positions = [i for start, end in spans for i in range(start, end)]
        masked = self.mask_positions(positions)
        return replace(masked, region_label=name)


@dataclass(frozen=True)
class MaskedAbInput:
    """An :class:`AbInput` with some positions replaced by ``<mask>``."""

    base: AbInput
    positions: Tuple[int, ...]
    region_label: Optional[str] = None

    def __len__(self) -> int:
        return len(self.base)


@dataclass(frozen=True)
class AgEmbedding:
    """One antigen's per-residue ESM-2 embedding -- antibody-*blind*, the
    raw input the model's antigen adapter consumes."""

    sequence: str
    embedding: torch.Tensor  # [L, dim] float32, cpu
    truncated: bool

    def __len__(self) -> int:
        return self.embedding.shape[0]


def encode_antibody_sequence(
    tokenizer: PreTrainedTokenizerBase, heavy: str, light: Optional[str] = None,
) -> AbInput:
    """Tokenize heavy(+light), no masking.

    Tokenized with ``add_special_tokens=False``: the frozen tower adds
    CLS/EOS itself and strips them on the way out (see ``model.py``'s
    ``_encode_antibody``), so the indices here stay aligned to the raw
    sequence and callers can pass spans in sequence coordinates.

    Args:
        tokenizer: The ESM-2 tokenizer, as held by :class:`~langaai.LangAAI`.
        heavy: Heavy-chain variable-domain sequence.
        light: Light-chain variable-domain sequence, if paired.
    """
    heavy_tok = np.array(tokenizer.encode(heavy, add_special_tokens=False), dtype=np.int64)
    tokens = [heavy_tok]
    segment = [np.full(len(heavy_tok), SEG_HEAVY, dtype=np.int8)]
    if light:
        light_tok = np.array(tokenizer.encode(light, add_special_tokens=False), dtype=np.int64)
        tokens.append(light_tok)
        segment.append(np.full(len(light_tok), SEG_LIGHT, dtype=np.int8))
    return AbInput(
        heavy=heavy, light=light,
        tokens=np.concatenate(tokens), segment=np.concatenate(segment),
    )


@torch.no_grad()
def embed_antigen(
    tower: EsmModel, tokenizer: PreTrainedTokenizerBase, device: torch.device,
    sequence: str, max_len: int = 4096,
) -> AgEmbedding:
    """A live single-sequence ESM-2 pass for a new antigen.

    ESM-2 attention is O(L^2), so ``max_len`` caps the cost by truncating
    the antigen's tail. Truncation is silent apart from the returned
    :attr:`AgEmbedding.truncated` flag -- check it if you care.

    Note that ``max_len``'s default is far above the joint-sequence length
    the model was trained at (:data:`TRAINED_MAX_JOINT_LEN` tokens, covering
    the antibody *and* the antigen together). Longer antigens still run, and
    the joint stack has no positional encoding of its own to extrapolate,
    but they are outside the training distribution; a warning is emitted.

    Args:
        tower: The frozen ESM-2 tower, i.e. ``model.model.ab_tower``.
        tokenizer: The matching ESM-2 tokenizer.
        device: Where to run the forward pass.
        sequence: The antigen's amino-acid sequence.
        max_len: Truncate the antigen beyond this many residues.

    Returns:
        An :class:`AgEmbedding` holding a ``[L, dim]`` float32 CPU tensor.
    """
    truncated = len(sequence) > max_len
    text = sequence[:max_len]
    if len(text) > TRAINED_MAX_JOINT_LEN - _TYPICAL_ANTIBODY_LEN:
        warnings.warn(
            f"Antigen is {len(text)} residues; combined with a typical "
            f"antibody this exceeds the {TRAINED_MAX_JOINT_LEN}-token joint "
            "sequence the model was trained on. It will run, but the result "
            "is outside the training distribution.",
            UserWarning,
            stacklevel=2,
        )
    encoded = tokenizer([text], return_tensors="pt", padding=False).to(device)
    hidden = tower(**encoded).last_hidden_state[0]
    # Strip CLS (position 0) and EOS (position len(text)+1); no padding at batch size 1.
    residues = hidden[1 : 1 + len(text)].to(torch.float32).cpu()
    return AgEmbedding(sequence=sequence, embedding=residues, truncated=truncated)
