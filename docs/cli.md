# Command-line interface

`predict` and `design` each read one JSON settings file:

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

Embeddings and attention are not CLI tasks: both return arrays rather than a
table of residues, and what you do with them is a Python question. Use
`model.cls_embedding(...)` and `model.attention_block(...)` directly — see
[examples/embed_pair.ipynb](../examples/embed_pair.ipynb) and
[examples/attention_map.ipynb](../examples/attention_map.ipynb).

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
  "pairs": [
    { "id": "example", "heavy": "...", "light": "...", "antigen": "...",
      "spans": [[96, 108]] }
  ]
}
```

A ready-to-run copy is in [examples/run.json](../examples/run.json).

`pairs` is the only required key.

### Shared keys

| Key | Default | Meaning |
|---|---|---|
| `device` | `"auto"` | `"auto"` picks CUDA, then Apple MPS, then CPU. Or name one: `"cpu"`, `"cuda"`, `"cuda:1"`, `"mps"`. |
| `checkpoint` | `null` | Path to the weights. `null` searches `LANGAAI_CHECKPOINT`, then `LANGAAI_CHECKPOINT_DIR`, then the package's own `checkpoints/`, then downloads them (see `langaai download`). |
| `top_k` | `5` | How many residues to report per position. `design` uses the top one; the rest are context. |
| `output` | `null` | JSON results file. `null` prints to stdout, so you can pipe into `jq`. |

### Per-pair keys

| Key | Required | Meaning |
|---|---|---|
| `heavy` | yes | Heavy-chain **variable domain** — not the full chain with its constant region. |
| `light` | no | Light-chain variable domain. Omit for a heavy-only antibody. |
| `antigen` | yes | Antigen sequence. |
| `spans` | yes | Half-open `[start, end)` index pairs, 0-based, into heavy followed by light. |
| `id` | no | Your own label for the pair, echoed into the output so you can match a result back to its input. The model never sees it. Must be unique. Defaults to `pair_0`, `pair_1`, … |

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
