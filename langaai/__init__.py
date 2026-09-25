"""LangAAI: antigen-conditioned masked language model for antibody CDRs.

``load()`` is the public entry point:

    >>> import langaai
    >>> model = langaai.load()
    >>> ab = model.encode_antibody(heavy="EVQLV...", light="DIQMT...")
    >>> ag = model.embed_antigen("MFVFLVLLPLVSSQC...")
    >>> masked = ab.mask_positions([10, 11, 12])
    >>> model.predict_masked([(masked, ag)])

``LangAAI`` (the object ``load()`` returns) is a thin, stateful convenience
wrapper: every method just forwards to a free function in ``predict.py``/
``embeddings.py``/``attention.py``/``inputs.py`` with this instance's
``model``/``tokenizer``/``device`` filled in. Those modules' functions are
importable directly (e.g. ``from langaai.predict import score_sequence``)
if you'd rather manage the model/tokenizer/device yourself.

Every pair-taking method accepts a *list* of ``(ab_input, ag_embedding)``
pairs -- pass a single-item list for one pair. There's no separate
"batch" variant: see ``_batching.py``'s module docstring for why each pair
gets its own forward pass internally rather than a padded tensor batch.

See MODEL_CARD.md for what the model was trained and evaluated on, and in
particular for what ``score_sequence`` is and is not validated for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from transformers import AutoTokenizer

from . import attention, checkpoint, embeddings, inputs, predict
from .embeddings import mean_pool
from .inputs import AbInput, AgEmbedding, MaskedAbInput
from .model import JointCDRMLM, ModelConfig
from .predict import PositionPrediction

__all__ = [
    "load", "LangAAI",
    "AbInput", "MaskedAbInput", "AgEmbedding", "PositionPrediction",
    "mean_pool", "ATTENTION_BLOCK_KINDS",
    "attention", "checkpoint", "embeddings", "inputs", "predict",
]

ATTENTION_BLOCK_KINDS = attention.ATTENTION_BLOCK_KINDS

Pair = Tuple[AbInput, AgEmbedding]
MaskedPair = Tuple[MaskedAbInput, AgEmbedding]


def _pick_device(requested: str) -> torch.device:
    """An explicit request wins; otherwise CUDA, then Apple MPS, then CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LangAAI:
    """A loaded LangAAI checkpoint: model + tokenizer + device, with every
    public function bound as a method."""

    def __init__(self, model: JointCDRMLM, tokenizer, device: torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dim = model.dim

    # --- inputs ---------------------------------------------------------
    def encode_antibody(self, heavy: str, light: Optional[str] = None) -> AbInput:
        return inputs.encode_antibody_sequence(self.tokenizer, heavy, light)

    def embed_antigen(self, sequence: str, max_len: int = 4096) -> AgEmbedding:
        return inputs.embed_antigen(self.model.ab_tower, self.tokenizer, self.device, sequence, max_len)

    # --- prediction -------------------------------------------------------
    def predict_masked(self, pairs: Sequence[MaskedPair], top_k: int = 5) -> List[List[PositionPrediction]]:
        return predict.predict_masked(self.model, self.tokenizer, self.device, pairs, top_k)

    def residue_probabilities(self, pairs: Sequence[MaskedPair]) -> List[Dict[int, Dict[str, float]]]:
        return predict.residue_probabilities(self.model, self.tokenizer, self.device, pairs)

    def score_sequence(
        self, pairs: Sequence[Pair], positions: Optional[Sequence[Optional[Sequence[int]]]] = None,
    ) -> List[float]:
        return predict.score_sequence(self.model, self.device, pairs, positions)

    # --- embeddings ---------------------------------------------------------
    def antibody_embedding_blind(self, ab_list: Sequence[AbInput]) -> List[torch.Tensor]:
        return embeddings.antibody_embedding_blind(self.model, self.device, ab_list)

    def antibody_embedding_conditioned(self, pairs: Sequence[Pair]) -> List[torch.Tensor]:
        return embeddings.antibody_embedding_conditioned(self.model, self.device, pairs)

    def antigen_embedding_blind(self, ag_list: Sequence[AgEmbedding]) -> List[torch.Tensor]:
        return embeddings.antigen_embedding_blind(ag_list)

    def antigen_embedding_conditioned(self, pairs: Sequence[Pair]) -> List[torch.Tensor]:
        return embeddings.antigen_embedding_conditioned(self.model, self.device, pairs)

    def cls_embedding(self, pairs: Sequence[Pair]) -> List[torch.Tensor]:
        return embeddings.cls_embedding(self.model, self.device, pairs)

    # --- attention -----------------------------------------------------
    def attentions(self, pairs: Sequence[Pair]) -> List[List[torch.Tensor]]:
        return attention.attentions(self.model, self.device, pairs)

    def attention_block(
        self, pairs: Sequence[Pair], kind: str,
        layer: Optional[int] = None, head: Optional[int] = None, average: bool = False,
    ) -> Union[List[torch.Tensor], List[List[torch.Tensor]]]:
        return attention.attention_block(self.model, self.device, pairs, kind, layer, head, average)


def load(device: str = "auto", checkpoint_path: Optional[Path] = None) -> LangAAI:
    """Load the model, ready for inference.

    The weights are downloaded from the Hugging Face Hub on first use and
    cached; subsequent calls are offline. See
    :mod:`langaai.checkpoint` for the full search order and for how to point
    this at a file you already have.

    Args:
        device: ``"auto"`` (CUDA > MPS > CPU), or an explicit
            ``torch.device``-compatible string (``"cpu"``, ``"cuda"``, ...).
        checkpoint_path: Load these weights instead of the standard ones.

    Returns:
        A :class:`LangAAI` in eval mode on the chosen device.
    """
    resolved_device = _pick_device(device)
    payload = checkpoint.load_state(checkpoint_path)
    model = JointCDRMLM(ModelConfig(**payload["config"]["model"]))
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(resolved_device).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.ENCODER)
    return LangAAI(model, tokenizer, resolved_device)
