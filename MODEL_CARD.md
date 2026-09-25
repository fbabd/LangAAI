# Model card: LangAAI

## What this is

**JointCDR-MLM**, an antigen-conditioned masked language model for antibody
CDR sequences: two ESM-2 towers (a frozen antibody tower run every forward
pass, and a per-residue antigen embedding fed in directly) meeting in a
shared, unrestricted joint self-attention stack. The question it's built to
answer: *does conditioning an antibody language model on its cognate
antigen's sequence measurably improve CDR prediction, beyond what
antibody-only language models and memorization already give you?*

The released checkpoint uses a `facebook/esm2_t12_35M_UR50D` backbone
(width 480) with a 4-layer, 8-head joint stack. Of its 48.8M parameters,
33.3M are the frozen ESM-2 tower and 15.5M were trained: the two per-side
adapters, the joint stack, the segment embeddings, the pooled token, and the
LM head (whose output projection is tied to the encoder's token embeddings).

## Training data

- 670,139 raw antibody-antigen rows pooled from 13 sources (patents,
  AlphaSeq, COVID-19, HIV, literature, plus smaller structural/benchmark
  sets), filtered to 427,665 rows with a resolvable heavy chain and known
  antibody/antigen clusters.
- **Entity-holdout split**: antibody and antigen *clusters* (not rows) are
  independently assigned to train/val/test, so a held-out antibody or
  antigen cluster never appears in training by construction — not by
  post-hoc dedup. Held-out rows are further split by which side is novel:
  `ab_novel` (n=8,551 test), `ag_novel` (n=8,804 test), `both_novel` (n=628
  test, an order of magnitude smaller — treat its numbers cautiously).
- **Known caveat**: ~44% of test CDR3 strings also appear verbatim in
  training. This is a real memorization confound for any raw-NLL number —
  see "How it was evaluated" below for how the project's own evaluation
  controls for it.

## How it was evaluated

The numbers below come from the accompanying paper; see it for full
methodology, confidence intervals and figures. **TODO: add the citation and
link once the paper is public.**

**Task**: recover masked CDR3 (and CDR1/CDR2/flank) residues on the
held-out test split, given the true paired antigen.

**Primary metric — paired delta**: `NLL(CDR3 | shuffled antigen) −
NLL(CDR3 | true antigen)`. This is exactly 0 in expectation for a model that
never uses the antigen (whatever it knows from framework→CDR3 correlation
cancels in the subtraction) — so a positive delta is evidence of genuine
antigen conditioning, not memorization, *specifically because* it isn't raw
NLL. LangAAI (small): **0.078 nats** on `ab_novel`, **0.038 nats** on
`ag_novel` (95% CI excludes 0 in both); every antigen-blind floor
(ESM-2/AntiBERTy/AbLang2) scores ≈0 delta, as required by construction. The
antigen-*aware* published baseline MINT scores 0.001 — the direct
apples-to-apples comparison against an existing antigen-aware architecture,
not just against antigen-blind floors.

**Top-1 CDR3 recovery accuracy** (secondary, more legible, not a
principled-null metric like delta): LangAAI 63.8% on `ab_novel`, vs.
matched-scale ESM-2's own MLM head 21.3%, AntiBERTy 48.3%, AbLang2 47.2%,
MINT 18.2%.

**Memorization check**: delta stays positive and its CI excludes 0 even
restricted to CDR3s **never seen during training** (0.075 nats on
`ab_novel`) — so the signal isn't purely a memorized-lookup artifact, though
delta is somewhat larger on seen CDR3s (0.087 nats), as expected.

**Interpretability**: the pooled
`cls` embedding shifts substantially between the true and a shuffled antigen
(cosine-distance 0.32–0.38 on `ab_novel`/`ag_novel`) and a linear probe on
it recovers antigen-cluster identity near-perfectly (1.00 accuracy vs. 0.05
chance over 20 clusters) — antigen identity is clearly legible in the pooled
representation, even though raw per-position attention *mass* barely shifts
between true/shuffled antigen (the mechanism isn't simply "attend harder to
the antigen when it's correct").

## `score_sequence`: what it is and isn't

`predict.score_sequence` / `LangAAI.score_sequence` reports a mean masked-
reconstruction log-likelihood — how well the joint stack predicts given
residues from the rest of the sequence plus the antigen. **It is not a
validated binding-affinity predictor.** Nothing in this project's evaluation
work established a connection between reconstruction likelihood and binding
affinity; treat a higher score as "more consistent with this model's
learned CDR-given-antigen distribution," not as "binds better" or "higher
affinity."

**What *is* validated for affinity** is a downstream regressor fitted on
`cls_embedding`: 5-fold cross-validated Ridge regression against three real
affinity-labeled datasets (ΔG, IC50, -log K_D). Headline
Pearson correlations (LangAAI-small `cls_embedding`, mean ± SEM over folds):
ΔG 0.563 ± 0.024, IC50 0.718 ± 0.027, -log K_D 0.930 ± 0.000 — comparable to,
and on -log K_D slightly behind, a simple ESM-2(antigen)+ESM-2(antibody)
concatenation baseline, and clearly ahead of the PPLM/MINT baselines
reported there. If you need affinity prediction, start from `cls_embedding`
plus your own fitted regressor, not from `score_sequence`.

## Known limitations

- **`both_novel` is underpowered** (628 test rows) — its confidence
  intervals are wide; don't read a crossing-zero CI there as evidence the
  antigen-conditioning effect vanishes for jointly-novel pairs, just that
  there isn't enough data yet to say either way.
- **No structural validation.** Everything here is sequence-only; nothing
  in this project checks predictions against solved structures or
  experimental binding assays beyond the affinity-regression check above.
- **CDR spans are always caller-supplied.** This package bundles no
  CDR-numbering tool, so it cannot tell you where CDR-H3 begins.
  `mask_region` and the CLI's `spans` both require you to supply spans from
  your own numbering step (ANARCI, IMGT/V-QUEST, or annotation your data
  already carries).
- **Antigen length cap, and a training-length caveat.** `embed_antigen`'s
  default `max_len=4096` truncates longer antigens (ESM-2 attention is
  O(L²)); truncation is silent unless you check `AgEmbedding.truncated`.
  Separately, the model was trained on joint sequences of at most 1,024
  tokens covering the antibody *and* antigen together, so antigens of more
  than roughly 780 residues are outside the training distribution even
  though they run. A warning is emitted in that case.
- **English-language docs, protein-sequence-only inputs.** No support for
  nucleotide sequences, non-standard/modified residues beyond the `X`
  ambiguity code, or multi-antigen complexes.
