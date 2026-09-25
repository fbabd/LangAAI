# Command-line interface

Every task reads one JSON settings file:

```bash
langaai <task> --config run.json
```

Start from a working example and edit it:

```bash
langaai schema > run.json
langaai design --config run.json
```

The file `langaai schema` prints is runnable as-is — it carries a real
antibody-antigen pair — and is annotated with `//` comments. Comments are
accepted anywhere in these files, including after a value, even though plain
JSON has none.

If you installed the package but `langaai` is not found, use
`python -m langaai.cli` instead; it is the same entry point.

## `langaai download`

The weights are fetched automatically the first time a model loads. This
command does it up front instead, which is what you want before a batch job,
or on a login node when the compute nodes have no network access.

```bash
langaai download                      # into the package's own checkpoints/
langaai download --dest /shared/models
langaai download --url https://.../langaai.pt
langaai download --force              # re-download over an existing file
```

It prints the saved path on stdout and exits `0`; on failure it prints the
reason on stderr and exits `1`. Downloads are checksum-verified and moved
into place atomically, so an interrupted run leaves nothing behind.

---

## The settings file

```json
{
  "device": "auto",
  "checkpoint": null,
  "output": null,
  "top_k": 5,
  "embeddings": ["cls"],
  "pool": true,
  "attention": { "kind": "ab_to_ag", "layer": null, "head": null,
                 "average": true, "top_n": 10 },
  "pairs": [
    { "id": "example", "heavy": "...", "light": "...", "antigen": "...",
      "spans": [[96, 108]] }
  ]
}
```

`pairs` is the only required key.

### Shared keys

| Key | Default | Meaning |
|---|---|---|
| `device` | `"auto"` | `"auto"` picks CUDA, then Apple MPS, then CPU. Or name one: `"cpu"`, `"cuda"`, `"cuda:1"`, `"mps"`. |
| `checkpoint` | `null` | Path to the weights. `null` searches `LANGAAI_CHECKPOINT`, then `LANGAAI_CHECKPOINT_DIR`, then the package's own `checkpoints/`, then downloads them (see `langaai download`). |
| `output` | `null` | Where results go. For `predict`/`design` this is a JSON file, and `null` means stdout. For `embed`/`attention` it is a `.npz` archive, defaulting to `langaai_embeddings.npz` / `langaai_attention.npz`. |

### Per-pair keys

| Key | Required | Meaning |
|---|---|---|
| `heavy` | yes | Heavy-chain **variable domain** — not the full chain with its constant region. |
| `light` | no | Light-chain variable domain. Omit for a heavy-only antibody. |
| `antigen` | yes | Antigen sequence. |
| `spans` | for `predict`, `design` | Half-open `[start, end)` index pairs, 0-based, into heavy followed by light. |
| `id` | no | Label used in the output and as the `.npz` key. Defaults to `pair_0`, `pair_1`, … |

Sequences may be lower-case (they are upper-cased) but must contain only the
20 amino acids plus `X`. Anything else is rejected by name.

**Span coordinates.** Indices run over the heavy chain then the light chain,
concatenated, with no offset — position `len(heavy)` is the light chain's
first residue. Tokenization is one token per residue, so these are ordinary
string indices. A span outside the chain is rejected before the model loads.

This package computes **no** CDR boundaries. Supply spans from your own
numbering step.

---

## `langaai predict`

Top-`top_k` residues at every position covered by `spans`, with
probabilities over the 20 amino acids plus `X`.

```bash
langaai predict --config run.json --top-k 3 --output predictions.json
```

```json
{
  "task": "predict",
  "top_k": 3,
  "results": [
    {
      "id": "example",
      "spans": [[96, 108]],
      "positions": [
        {
          "position": 96,
          "native": "A",
          "top": [
            { "residue": "A", "probability": 0.712 },
            { "residue": "T", "probability": 0.139 },
            { "residue": "S", "probability": 0.080 }
          ]
        }
      ]
    }
  ]
}
```

`native` is the residue actually in your input at that position, so you can
score the prediction without re-deriving the index yourself.

All positions in `spans` are masked in a **single** forward pass, so each
position's distribution is conditioned on the framework and the antigen but
**not** on the other masked positions.

---

## `langaai design`

Masks whole spans and reads off the most probable residue at each position —
one complete designed loop.

```bash
langaai design --config run.json --output design.json
```

```json
{
  "task": "design",
  "results": [
    {
      "id": "example",
      "spans": [[96, 108]],
      "native": "ARWGTVEWFFDY",
      "designed": "ARGDYYYYWFDY",
      "recovery": 0.416667,
      "n_positions": 12,
      "positions": [
        { "position": 96, "native": "A", "designed": "A",
          "probability": 0.712, "match": true }
      ]
    }
  ]
}
```

`recovery` is amino-acid recovery: the fraction of positions where the
design matches what was already there. It is a *recovery* number, not a
quality number — a design that recovers nothing may still be a fine binder,
and a high-recovery design is not automatically one.

**This is not a joint sample.** Because every position is masked in one
forward pass, taking the arg-max independently at each position can produce
a combination the model would never rank first as a whole sequence. To
search that space properly — and to see how much the ranking moves once
positions condition on each other — see
[examples/candidate_cdr3_search.ipynb](../examples/candidate_cdr3_search.ipynb).

`spans` are not required to hold a native loop; whatever is there is
reported as `native` and used as the recovery baseline.

---

## `langaai embed`

Writes representations to a `.npz` archive, keyed `"<pair id>/<kind>"`, and
prints a JSON manifest to stdout.

```bash
langaai embed --config run.json --output vectors.npz
```

```json
{
  "task": "embed",
  "archive": "vectors.npz",
  "pooled": true,
  "embeddings": ["cls", "antibody_conditioned"],
  "keys": ["example/antibody_conditioned", "example/cls"],
  "results": [
    {
      "id": "example",
      "antigen_truncated": false,
      "n_antibody_residues": 232,
      "n_antigen_residues": 183,
      "shapes": { "cls": [480], "antibody_conditioned": [480] }
    }
  ]
}
```

| Kind | Shape (unpooled) | What it is |
|---|---|---|
| `cls` | `[dim]` | The native pair-level pooled vector. Start here for a downstream predictor. |
| `antibody_blind` | `[n_ab, dim]` | Frozen ESM-2 over the antibody alone; never sees the antigen. |
| `antibody_conditioned` | `[n_ab, dim]` | Antibody after the joint stack — antigen-aware. |
| `antigen_blind` | `[n_ag, dim]` | The raw antigen embedding fed into the model. |
| `antigen_conditioned` | `[n_ag, dim]` | Antigen after the joint stack — "paratope-aware". |

`"pool": true` (the default) mean-pools the per-residue kinds down to one
`[dim]` vector each. `cls` is already one vector, so pooling leaves it
alone. Set `"pool": false` to keep per-residue rows.

Reading it back:

```python
import numpy as np
archive = np.load("vectors.npz")
vector = archive["example/cls"]          # [480]
```

The manifest reports `antigen_truncated` per pair — check it if any antigen
might exceed `max_len`.

---

## `langaai attention`

Extracts one segment slice of the joint attention matrix to a `.npz`, keyed
by pair id, and prints a JSON summary.

```bash
langaai attention --config run.json --output attention.npz
```

| `attention` key | Default | Meaning |
|---|---|---|
| `kind` | `"ab_to_ag"` | Which slice — see the table below. |
| `layer` | `null` | Layer index, or `null` for all layers. |
| `head` | `null` | Head index, or `null` for all heads. |
| `average` | `true` | Average over whichever of `layer`/`head` was left `null`. |
| `top_n` | `10` | How many top-attended positions to report. |

| `kind` | Rows → columns |
|---|---|
| `ab_to_ag` | antibody → antigen (paratope → epitope) |
| `ag_to_ab` | antigen → antibody (epitope → paratope) |
| `self_ab` | antibody → antibody |
| `self_ag` | antigen → antigen |
| `cls_to_ab` | pooled token → antibody |
| `cls_to_ag` | pooled token → antigen |

```json
{
  "task": "attention",
  "kind": "ab_to_ag",
  "results": [
    { "id": "example", "shape": [232, 183],
      "top_attended_positions": [122, 100, 178, 179, 98] }
  ]
}
```

`top_attended_positions` indexes the **column** side — for `ab_to_ag`, the
antigen positions drawing the most attention, summed over antibody
positions. Attention mass is not a contact prediction; see
[MODEL_CARD.md](../MODEL_CARD.md).

The task needs a single matrix per pair, so either set `layer` to an integer
or leave `average` as `true`. Asking for all layers unaveraged is rejected
up front.

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. |
| `1` | The model could not be loaded — weights missing, or `transformers` too old. |
| `2` | The settings file is unusable, or no task was given. Nothing ran. |

Settings are validated **before** the weights are loaded, so a typo fails in
seconds rather than after a download.

---

## Long antigens

The model was trained on joint sequences of at most **1,024 tokens**,
covering the antibody *and* the antigen together. `embed_antigen` truncates
at 4,096 residues by default and warns above roughly 780, because beyond
that the input is outside the training distribution even though it runs.
Treat those results with suspicion.
