"""JointCDR-MLM: two frozen ESM-2 towers into a late joint self-attention stack.

The antibody tower runs inside the forward pass on token-level-masked input. It
is frozen, so this is forward-only over ~120 tokens and cheap. It cannot be
replaced by a precomputed cache: the input it sees is already masked, so a
cache keyed on the unmasked sequence would leak the residues being predicted.

Contents:
    ModelConfig: Hyperparameter dataclass (encoder, stack depth/heads, unfreeze_top_k).
    JointCDRMLM: The model.
        _encode_antibody: Frozen-tower forward pass, CLS/EOS add-then-strip.
        forward: Encode both sides, join with segment embeddings, run the
            joint stack, decode antibody-slice logits (optionally with
            attentions/hidden states via return_dict).
        loss: Masked cross-entropy over labelled positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel

from .adapters import SideAdapter
from .joint_attention import JointStack, StackConfig
from .lm_head import TiedLMHead

__all__ = ["JointCDRMLM", "ModelConfig"]

IGNORE_INDEX = -100
N_SEGMENTS = 4


@dataclass(frozen=True)
class ModelConfig:
    """Model hyperparameters.

    Attributes:
        encoder: HuggingFace id, used for both towers.
        n_layers: Joint stack depth.
        n_heads: Attention heads.
        dropout: Dropout in the joint stack.
        unfreeze_top_k: Antibody tower layers left trainable, counted from the
            top. Zero keeps the tower fully frozen, which is what the released
            checkpoint uses; it is retained here because the checkpoint's
            stored config sets it, and because the field changes which
            parameters require grad.
    """

    encoder: str = "facebook/esm2_t12_35M_UR50D"
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    unfreeze_top_k: int = 0


class JointCDRMLM(nn.Module):
    """Antigen-conditioned CDR masked language model."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ab_tower = EsmModel.from_pretrained(cfg.encoder, add_pooling_layer=False)
        self.dim = int(self.ab_tower.config.hidden_size)
        self.vocab_size = int(self.ab_tower.config.vocab_size)

        tokenizer = AutoTokenizer.from_pretrained(cfg.encoder)
        self.cls_token_id = int(tokenizer.cls_token_id)
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.mask_token_id = int(tokenizer.mask_token_id)

        self.ab_tower.requires_grad_(False)
        if cfg.unfreeze_top_k > 0:
            for layer in self.ab_tower.encoder.layer[-cfg.unfreeze_top_k :]:
                layer.requires_grad_(True)

        self.token_embedding = self.ab_tower.embeddings.word_embeddings
        self.ab_adapter = SideAdapter(self.dim)
        self.ag_adapter = SideAdapter(self.dim)
        self.segment_emb = nn.Embedding(N_SEGMENTS, self.dim)
        nn.init.zeros_(self.segment_emb.weight)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))

        self.stack = JointStack(
            StackConfig(self.dim, cfg.n_layers, cfg.n_heads, cfg.dropout)
        )
        self.head = TiedLMHead(self.dim, self.token_embedding)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _encode_antibody(self, tokens: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """Run the frozen tower, adding and then stripping CLS/EOS.

        The tower is a single-chain model, so special tokens are added here and
        removed on the way out. Region labels therefore stay aligned to the raw
        chain in exactly one place.
        """
        batch, length = tokens.shape
        cls = torch.full((batch, 1), self.cls_token_id, device=tokens.device)
        eos = torch.full((batch, 1), self.eos_token_id, device=tokens.device)
        padded = torch.cat([cls, tokens.masked_fill(pad, self.pad_token_id), eos], dim=1)
        attention = torch.cat([
            torch.ones(batch, 1, device=tokens.device, dtype=torch.long),
            (~pad).long(),
            torch.ones(batch, 1, device=tokens.device, dtype=torch.long),
        ], dim=1)

        grad = torch.enable_grad if self.cfg.unfreeze_top_k > 0 else torch.no_grad
        with grad():
            hidden = self.ab_tower(
                input_ids=padded, attention_mask=attention
            ).last_hidden_state
        return hidden[:, 1 : 1 + length]

    def forward(
        self,
        ab_tokens: torch.Tensor,
        ab_pad: torch.Tensor,
        ag_emb: torch.Tensor,
        ag_pad: torch.Tensor,
        segment_ids: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Predict antibody tokens from the joint sequence.

        Args:
            ab_tokens: ``[B, L_ab]`` corrupted antibody token ids.
            ab_pad: ``[B, L_ab]`` bool, ``True`` at pad.
            ag_emb: ``[B, L_ag, dim]`` cached antigen embeddings.
            ag_pad: ``[B, L_ag]`` bool, ``True`` at pad.
            segment_ids: ``[B, 1 + L_ab + L_ag]``.
            return_dict: if True, also return the joint stack's attention
                matrices and its contextualised cls/ab/ag embedding slices.

        Returns:
            By default, ``[B, L_ab, vocab]`` logits over the antibody slice
            only. If ``return_dict``, a dict with keys ``logits``,
            ``attentions`` (one ``[B, n_heads, L, L]`` tensor per joint-stack
            layer), ``cls`` (``[B, 1, dim]``), ``ab`` (``[B, L_ab, dim]``),
            and ``ag`` (``[B, L_ag, dim]``).
        """
        batch, n_ab = ab_tokens.shape
        ab = self.ab_adapter(self._encode_antibody(ab_tokens, ab_pad))
        ag = self.ag_adapter(ag_emb)

        cls = self.cls_token.expand(batch, -1, -1)
        joint = torch.cat([cls, ab, ag], dim=1) + self.segment_emb(segment_ids)

        pad = torch.cat([
            torch.zeros(batch, 1, dtype=torch.bool, device=ab_pad.device), ab_pad, ag_pad,
        ], dim=1)

        if not return_dict:
            hidden = self.stack(joint, pad)
            return self.head(hidden[:, 1 : 1 + n_ab])

        hidden, attentions = self.stack(joint, pad, need_weights=True)
        return {
            "logits": self.head(hidden[:, 1 : 1 + n_ab]),
            "attentions": attentions,
            "cls": hidden[:, :1],
            "ab": hidden[:, 1 : 1 + n_ab],
            "ag": hidden[:, 1 + n_ab :],
        }

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy over labelled positions only."""
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
