# Examples

Five notebooks, in the order they build on each other. Each one runs
standalone and downloads the weights on first use.

```bash
pip install -e ".[notebooks]"   # or: pip install langaai matplotlib jupyter
jupyter lab
```

Outputs are not checked in, so every notebook starts blank — run the cells
to see results. On CPU each notebook takes well under a minute, apart from
`candidate_cdr3_search.ipynb`, which runs one forward pass per loop position.

| Notebook | What it shows |
|---|---|
| **`predict_cdr3.ipynb`** | The basics: load the model, mask a known CDR3, read the top-5 residues per position. Start here. |
| **`design_cdr3.ipynb`** | Mask the **whole** loop at once and read off one complete designed CDR-H3, then score it by amino-acid recovery. |
| **`candidate_cdr3_search.ipynb`** | Generate many candidate loops, rank them, and see why the obvious score is the wrong one. The most useful notebook if you plan to design sequences. |
| **`embed_pair.ipynb`** | All five embedding flavours for one pair, and how they differ. |
| **`attention_map.ipynb`** | Where CDR3 attention lands on the antigen. Needs `matplotlib`. |

## If you read only one thing

`candidate_cdr3_search.ipynb` demonstrates a trap worth knowing before you
use this model for design. `score_sequence` masks the positions it scores,
so scoring a candidate over a fully-masked span gives the *same* number no
matter which residues the candidate carries — it cannot tell candidates
apart. The notebook shows the score that can, by masking one position at a
time so a candidate's residues condition on each other, and shows that the
two rankings genuinely disagree.

## The same things from the command line

Every notebook except `candidate_cdr3_search.ipynb` has a CLI equivalent
that takes a JSON settings file — see [../docs/cli.md](../docs/cli.md).

| Notebook | Command |
|---|---|
| `predict_cdr3.ipynb` | `langaai predict --config run.json` |
| `design_cdr3.ipynb` | `langaai design --config run.json` |
| `embed_pair.ipynb` | `langaai embed --config run.json` |
| `attention_map.ipynb` | `langaai attention --config run.json` |

## Spans

Every notebook supplies CDR spans explicitly, because this package bundles
no CDR-numbering tool and cannot derive them from a raw sequence. Where a
notebook finds a span by substring search, that is a convenience for the
example only — in real use, spans come from your own numbering step.
