"""Command-line interface: ``langaai <task> --config <file.json>``.

Two tasks read one JSON settings file describing the antibody-antigen pairs
to run and where to put the results, so a run is reproducible from a file
you can keep beside its output rather than from shell history.

    langaai predict  --config run.json
    langaai design   --config run.json
    langaai schema                       # print an annotated example config
    langaai download                     # fetch the weights ahead of time

Embeddings and attention are deliberately not exposed here: both return
arrays rather than a table of residues, and what you do with them is a
Python question. Use :func:`langaai.LangAAI.cls_embedding` and
:func:`langaai.LangAAI.attention_block` directly -- see ``examples/``.

``langaai schema`` is the reference for the settings file; docs/cli.md walks
through each task.

Contents:
    main: Entry point; dispatches on the subcommand.
    load_config: Read, parse and validate a settings file.
    Config/PairSpec: The validated settings.
    cmd_predict/cmd_design: One per task.
    cmd_download: Fetch the weights without running anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["main", "load_config", "Config", "PairSpec"]

_VALID_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYX")

EXAMPLE_CONFIG = """\
{
  // Shared settings. Every key here is optional and has the default shown.
  "device": "auto",          // "auto" (CUDA > MPS > CPU), "cpu", "cuda", "mps"
  "checkpoint": null,        // path to weights; null = download automatically
  "output": null,            // JSON results file; null = stdout

  // How many residues to report per position.
  "top_k": 5,

  // The pairs to run. One entry per antibody-antigen pair. The pair below
  // is a real one, so this file runs as-is:
  //     langaai schema > run.json && langaai design --config run.json
  "pairs": [
    {
      // Optional label, echoed into the output so you can match results to
      // inputs. Must be unique. Defaults to pair_0, pair_1, ...
      "id": "example",

      // Variable domains only -- not the constant region.
      "heavy": "QVQLQQPGAELVRPGASVKLSCKASGYTFTSYWMNWVKQRPEQGLEWIGRIDPYDSETHYNQKFKDKAILTVDKSSTTAYMQLSSLTSEDSAVYYCARWGTVEWFFDYWGQGTTLTVSQ",

      // Optional. Omit for a heavy-chain-only antibody.
      "light": "DIVMTQSPSSLAMSVGQKVTMSCKSSQSLLNSYNQENYLAWYQQKPGQSPKLLVYFASTRESGVPDRFIGSGSGTDFTLTISSVQAEDLADYFCQQHYSTPFTFGSGTKLEIK",

      "antigen": "CPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGTIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGSKPCNGVKGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELL",

      // Half-open [start, end) index pairs, 0-based, into heavy followed by
      // light. Required. This one is the CDR-H3 loop.
      "spans": [[96, 108]]
    }
  ]
}
"""


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


class ConfigError(Exception):
    """A settings file that cannot be used as given."""


@dataclass(frozen=True)
class PairSpec:
    """One antibody-antigen pair from the settings file."""

    id: str
    heavy: str
    light: Optional[str]
    antigen: str
    spans: Tuple[Tuple[int, int], ...]

    @property
    def positions(self) -> List[int]:
        """Every index covered by :attr:`spans`, ascending and deduplicated."""
        return sorted({i for start, end in self.spans for i in range(start, end)})


@dataclass(frozen=True)
class Config:
    """Validated settings for one CLI run."""

    pairs: Tuple[PairSpec, ...]
    device: str = "auto"
    checkpoint: Optional[str] = None
    output: Optional[str] = None
    top_k: int = 5


def _strip_comments(text: str) -> str:
    """Drop ``//`` comments so a hand-annotated settings file still parses.

    JSON has no comments, but these files are meant to be edited by hand and
    annotated, so ``//`` to end-of-line is accepted -- including after a
    value. Tracks string state, so a ``//`` inside a quoted value (a URL or
    a path) is left alone.
    """
    out = []
    in_string = False
    escaped = False
    i = 0
    while i < len(text):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            out.append(char)
        elif char == '"':
            in_string = True
            out.append(char)
        elif char == "/" and text[i + 1: i + 2] == "/":
            newline = text.find("\n", i)
            if newline == -1:
                break
            i = newline
            continue
        else:
            out.append(char)
        i += 1
    return "".join(out)


def _require_sequence(value: Any, what: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where}: {what!r} must be a non-empty string")
    seq = value.strip().upper()
    bad = sorted(set(seq) - _VALID_RESIDUES)
    if bad:
        raise ConfigError(
            f"{where}: {what!r} contains characters that are not amino acids: "
            f"{''.join(bad)}. Expected only {''.join(sorted(_VALID_RESIDUES))}."
        )
    return seq


def _parse_spans(raw: Any, chain_len: int, where: str) -> Tuple[Tuple[int, int], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ConfigError(f"{where}: 'spans' must be a list of [start, end] pairs")

    spans = []
    for item in raw:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise ConfigError(
                f"{where}: each span must be a [start, end] pair, got {item!r}"
            )
        start, end = item
        if not (isinstance(start, int) and isinstance(end, int)):
            raise ConfigError(f"{where}: span bounds must be integers, got {item!r}")
        if not 0 <= start < end <= chain_len:
            raise ConfigError(
                f"{where}: span [{start}, {end}) is out of range for a "
                f"{chain_len}-residue antibody (need 0 <= start < end <= {chain_len})"
            )
        spans.append((start, end))
    return tuple(spans)


def _parse_pair(raw: Any, index: int) -> PairSpec:
    where = f"pairs[{index}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: must be an object, got {type(raw).__name__}")

    pair_id = str(raw.get("id") or f"pair_{index}")
    heavy = _require_sequence(raw.get("heavy"), "heavy", where)
    light_raw = raw.get("light")
    light = _require_sequence(light_raw, "light", where) if light_raw else None
    antigen = _require_sequence(raw.get("antigen"), "antigen", where)

    chain_len = len(heavy) + (len(light) if light else 0)
    spans = _parse_spans(raw.get("spans"), chain_len, where)
    return PairSpec(id=pair_id, heavy=heavy, light=light, antigen=antigen, spans=spans)


def load_config(path: Path) -> Config:
    """Read and validate a settings file.

    Args:
        path: The JSON settings file.

    Returns:
        A validated :class:`Config`.

    Raises:
        ConfigError: If the file is missing, is not valid JSON, or describes
            something that cannot be run.
    """
    if not path.exists():
        raise ConfigError(f"No settings file at {path}")
    try:
        raw = json.loads(_strip_comments(path.read_text()))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a JSON object")

    pairs_raw = raw.get("pairs")
    if not isinstance(pairs_raw, list) or not pairs_raw:
        raise ConfigError(
            f"{path}: 'pairs' must be a non-empty list. Run `langaai schema` "
            "for an example."
        )

    top_k = raw.get("top_k", 5)
    if not isinstance(top_k, int) or top_k < 1:
        raise ConfigError(f"{path}: 'top_k' must be a positive integer, got {top_k!r}")

    pairs = tuple(_parse_pair(p, i) for i, p in enumerate(pairs_raw))

    # Ids label the results, so a repeat makes the output ambiguous.
    seen: Dict[str, int] = {}
    for position, pair in enumerate(pairs):
        if pair.id in seen:
            raise ConfigError(
                f"{path}: duplicate pair id {pair.id!r} (pairs[{seen[pair.id]}] "
                f"and pairs[{position}]). Ids must be unique -- they are what "
                "matches a result back to its input."
            )
        seen[pair.id] = position

    return Config(
        pairs=pairs,
        device=raw.get("device", "auto"),
        checkpoint=raw.get("checkpoint"),
        output=raw.get("output"),
        top_k=top_k,
    )


def preflight(cfg: Config, task: str) -> None:
    """Validate task-specific settings before the model is loaded.

    Loading the model takes seconds and downloads weights on a cold cache,
    so anything knowable from the settings file alone is rejected first.
    """
    missing = [p.id for p in cfg.pairs if not p.spans]
    if missing:
        raise ConfigError(
            f"'{task}' needs a 'spans' entry for every pair; missing on: "
            f"{', '.join(missing)}. This package does not compute CDR "
            "boundaries -- supply them from your own numbering step."
        )


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------


def _build(model, pair: PairSpec):
    """Encode one pair's antibody and embed its antigen."""
    ab = model.encode_antibody(pair.heavy, pair.light)
    ag = model.embed_antigen(pair.antigen)
    return ab, ag


def _chain(pair: PairSpec) -> str:
    """The heavy+light string the model's position indices run over."""
    return pair.heavy + (pair.light or "")


def _emit(payload: Dict[str, Any], output: Optional[str]) -> None:
    text = json.dumps(payload, indent=2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
        print(f"wrote {path}", file=sys.stderr)
    else:
        print(text)


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------


def cmd_predict(model, cfg: Config) -> Dict[str, Any]:
    """Top-k residues at each position in the given spans."""
    results = []
    for pair in cfg.pairs:
        ab, ag = _build(model, pair)
        masked = ab.mask_region("spans", spans=list(pair.spans))
        [predictions] = model.predict_masked([(masked, ag)], top_k=cfg.top_k)
        chain = _chain(pair)
        results.append({
            "id": pair.id,
            "spans": [list(s) for s in pair.spans],
            "positions": [
                {
                    "position": p.position,
                    "native": chain[p.position],
                    "top": [{"residue": a, "probability": round(v, 6)} for a, v in p.top],
                }
                for p in predictions
            ],
        })
    return {"task": "predict", "top_k": cfg.top_k, "results": results}


def cmd_design(model, cfg: Config) -> Dict[str, Any]:
    """Mask whole spans and read off the arg-max residue at each position.

    All positions are masked in one forward pass, so the model's
    distribution over the span factorises and this is not a joint sample --
    see docs/cli.md and examples/candidate_cdr3_search.ipynb.
    """
    results = []
    for pair in cfg.pairs:
        ab, ag = _build(model, pair)
        masked = ab.mask_region("spans", spans=list(pair.spans))
        [predictions] = model.predict_masked([(masked, ag)], top_k=cfg.top_k)

        chain = _chain(pair)
        designed = "".join(p.top[0][0] for p in predictions)
        native = "".join(chain[p.position] for p in predictions)
        matches = [d == n for d, n in zip(designed, native)]

        results.append({
            "id": pair.id,
            "spans": [list(s) for s in pair.spans],
            "native": native,
            "designed": designed,
            "recovery": round(sum(matches) / len(matches), 6) if matches else None,
            "n_positions": len(matches),
            "positions": [
                {
                    "position": p.position,
                    "native": chain[p.position],
                    "designed": p.top[0][0],
                    "probability": round(p.top[0][1], 6),
                    "match": p.top[0][0] == chain[p.position],
                }
                for p in predictions
            ],
        })
    return {"task": "design", "results": results}


_TASKS = {
    "predict": cmd_predict,
    "design": cmd_design,
}


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="langaai",
        description=(
            "Antigen-conditioned masked language model for antibody CDRs. "
            "Each task is driven by a JSON settings file; run "
            "`langaai schema` for an annotated example."
        ),
    )
    sub = parser.add_subparsers(dest="task", metavar="<task>")

    descriptions = {
        "predict": "Top-k residues at each position of the given spans.",
        "design": "Mask whole spans and read off one designed sequence.",
    }
    for name, help_text in descriptions.items():
        task = sub.add_parser(name, help=help_text, description=help_text)
        task.add_argument(
            "--config", required=True, type=Path,
            help="JSON settings file (see `langaai schema`)",
        )
        task.add_argument("--output", help="Override the config's 'output'")
        task.add_argument("--device", help="Override the config's 'device'")
        task.add_argument("--checkpoint", help="Override the config's 'checkpoint'")
        task.add_argument("--top-k", type=int, help="Override the config's 'top_k'")

    sub.add_parser(
        "schema",
        help="Print an annotated example settings file.",
        description="Print an annotated example settings file.",
    )

    fetch = sub.add_parser(
        "download",
        help="Download the model weights now, instead of on first use.",
        description=(
            "Download the model weights and save them, so the first real run "
            "does not have to. Prints where they landed. Without this, the "
            "weights are fetched automatically the first time a model loads."
        ),
    )
    fetch.add_argument("--dest", type=Path, help="Directory to save into")
    fetch.add_argument("--url", help="Fetch from this URL instead of the default")
    fetch.add_argument(
        "--force", action="store_true", help="Re-download even if already present",
    )
    return parser


def cmd_download(args: argparse.Namespace) -> int:
    """Fetch the weights up front and report where they are."""
    from . import checkpoint

    try:
        path = checkpoint.download(
            dest_dir=args.dest, url=args.url, force=args.force,
        )
    except RuntimeError as exc:
        print(f"langaai download: {exc}", file=sys.stderr)
        return 1
    size = path.stat().st_size / 1e6
    print(f"{path}  ({size:.0f} MB)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.task is None:
        parser.print_help()
        return 2
    if args.task == "schema":
        print(EXAMPLE_CONFIG, end="")
        return 0
    if args.task == "download":
        return cmd_download(args)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"langaai {args.task}: {exc}", file=sys.stderr)
        return 2

    overrides: Dict[str, Any] = {}
    for key in ("output", "device", "checkpoint"):
        if getattr(args, key, None) is not None:
            overrides[key] = getattr(args, key)
    if getattr(args, "top_k", None) is not None:
        overrides["top_k"] = args.top_k
    if overrides:
        cfg = Config(**{**cfg.__dict__, **overrides})

    try:
        preflight(cfg, args.task)
    except ConfigError as exc:
        print(f"langaai {args.task}: {exc}", file=sys.stderr)
        return 2

    import langaai

    try:
        model = langaai.load(device=cfg.device, checkpoint_path=cfg.checkpoint)
    except (FileNotFoundError, ImportError) as exc:
        print(f"langaai {args.task}: {exc}", file=sys.stderr)
        return 1

    payload = _TASKS[args.task](model, cfg)
    _emit(payload, cfg.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
