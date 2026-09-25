# LangAAI

An **antigen-conditioned masked language model for antibody CDR sequences**.

Most antibody language models read the antibody alone. LangAAI reads the
antibody *and* its target antigen in one joint attention stack, so the
residues it predicts for a CDR loop depend on what that antibody is meant to
bind. Give it a heavy (and optionally light) variable domain plus an antigen
sequence, and it will:

- **predict** the residues at positions you mask, with probabilities;
- **design** a whole CDR loop from the framework and the antigen alone;
- **embed** the pair, as five distinct representations, for downstream models;
- **expose attention** between antibody and antigen positions.

It is sequence-only — it never sees a structure.



## Install

The weights are **not** in this repository — they are ~195 MB. They are
downloaded automatically the first time you load the model. Every later run reads that file directly, offline. You do not have to do anything.

### Option 1 — install the package (recommended)

```bash
pip install git+https://github.com/fbabd/LangAAI.git
```

That pulls in `torch`, `transformers`, `huggingface-hub` and `numpy`, and
puts a `langaai` command on your PATH. If you need a specific CUDA build of
PyTorch, install that first, from [pytorch.org](https://pytorch.org), then
run the line above.

### Option 2 — work in a clone

```bash
git clone https://github.com/fbabd/LangAAI.git
cd langaai
pip install -e ".[notebooks]"
```

`-e` means edits to the source take effect without reinstalling. The
`notebooks` extra adds `matplotlib` and `jupyter` for `examples/`.

### The weights

Nothing to do — the first `langaai.load()` fetches them:

```
Downloading LangAAI weights from https://huggingface.co/faisalashraf/langaai/resolve/main/langaai.pt
  to .../site-packages/langaai/checkpoints/langaai.pt  (about 195 MB, one time)
  downloading langaai.pt  [########################################] 100.0%  195/195 MB
Saved .../site-packages/langaai/checkpoints/langaai.pt
```

To fetch them ahead of time instead — useful before a batch job, or on a
login node with network access:

```bash
langaai download                 # prints where they landed
langaai download --dest /shared/models
langaai download --force         # re-download
```

The download is streamed to a temporary file, checked against a SHA-256, and
only then moved into place, so an interrupted download can never leave a
half-written checkpoint behind.


> **Requires `transformers>=5.0`.** The checkpoint stores ESM-2's rotary
> embeddings in the layout `transformers` 5.0 introduced. On 4.x, loading
> fails with missing keys for every layer's `rotary_embeddings.inv_freq`;
> `langaai` checks the version up front and tells you so.

## Quickstart — Python

```python
import langaai

model = langaai.load()          # CUDA > MPS > CPU, chosen automatically

ab = model.encode_antibody(heavy="QVQLQQPGAEL...", light="DIVMTQSPSSL...")
ag = model.embed_antigen("CPFGEVFNATRFASVY...")

# Mask the CDR-H3 loop and see what the model puts there.
# Spans are always yours to supply -- see "CDR spans" below.
masked = ab.mask_region("cdr3", spans=[(96, 108)])
[predictions] = model.predict_masked([(masked, ag)], top_k=5)

for p in predictions:
    print(p.position, p.top)    # [('W', 0.44), ('L', 0.18), ...]

# One pooled vector per pair -- the input for a downstream regressor.
[vector] = model.cls_embedding([(ab, ag)])
```

Every pair-taking method takes a **list** of `(antibody, antigen)` pairs;
pass a one-item list for a single pair. There is no separate batch API.

## Quickstart — command line

Every task is driven by one JSON settings file, so a run is reproducible
from a file you keep beside its output.

```bash
langaai schema > run.json     # an annotated, immediately runnable example
langaai design --config run.json
```

A ready-to-run settings file is also checked in at
[examples/run.json](examples/run.json):

```bash
langaai design  --config examples/run.json
langaai predict --config examples/run.json --top-k 3
```

```
langaai predict    Top-k residues at each masked position
langaai design     Mask whole spans, read off one designed sequence
langaai schema     Print an annotated example settings file
langaai download   Fetch the weights ahead of time
```

Embeddings and attention are Python-only — they return arrays rather than a
table of residues, so what you do with them is a Python question. See
[examples/](examples/).

**[docs/cli.md](docs/cli.md)** documents every settings key and both tasks'
output format.

## CDR spans

This package bundles **no CDR-numbering tool** and cannot work out where
CDR-H3 starts from a raw sequence. Every span is caller-supplied, as
half-open `(start, end)` indices into the heavy chain followed by the light
chain, 0-based. Get them from your own numbering step — ANARCI, IMGT/V-QUEST,
or whatever annotation your data already carries.

## API summary

| | |
|---|---|
| `langaai.load(device="auto", checkpoint_path=None)` | Load the model |
| `model.encode_antibody(heavy, light=None)` | Tokenize a heavy(+light) chain |
| `model.embed_antigen(sequence, max_len=4096)` | ESM-2 embedding of an antigen |
| `ab.mask_positions(indices)` / `ab.mask_region(name, spans)` | Mask positions |
| `model.predict_masked(pairs, top_k=5)` | Top-k residues per masked position |
| `model.residue_probabilities(pairs)` | Full distribution per masked position |
| `model.score_sequence(pairs, positions=None)` | Masked-reconstruction log-likelihood — **not an affinity predictor** |
| `model.antibody_embedding_blind/_conditioned(...)` | Antibody embedding, without / with antigen context |
| `model.antigen_embedding_blind/_conditioned(...)` | Antigen embedding, without / with antibody context |
| `model.cls_embedding(pairs)` | Pair-level pooled vector |
| `langaai.mean_pool(embedding)` | `[L, dim] -> [dim]` |
| `model.attentions(pairs)` | Raw per-layer attention, `[n_heads, L, L]` |
| `model.attention_block(pairs, kind, ...)` | Attention sliced by segment (`self_ab`, `self_ag`, `ab_to_ag`, `ag_to_ab`, `cls_to_ab`, `cls_to_ag`) |

## Examples

Runnable notebooks in **[examples/](examples/)** — start with
`predict_cdr3.ipynb`. See [examples/README.md](examples/README.md) for what
each one covers.

## Citation

**TODO: add the paper citation and BibTeX entry once it is public.**

