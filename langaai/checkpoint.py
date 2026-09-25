"""Finding, downloading and loading the LangAAI weights.

The weights are not shipped inside the package -- the checkpoint is ~195 MB,
larger than a package index or a git repository should carry. They are
hosted at :data:`CHECKPOINT_URL` (any ordinary HTTPS URL: a Hugging Face
model repository, an S3 bucket, a Zenodo record, a lab server) and are
downloaded **once**, on first use, into the package's own ``checkpoints``
directory. Every later call reads that file directly, offline.

Nothing about this is Hugging Face specific. To serve the weights from
somewhere else, set ``LANGAAI_CHECKPOINT_URL``, or edit
:data:`CHECKPOINT_URL` in a fork.

:func:`resolve` looks in four places, in order, so that an offline or
air-gapped machine is never forced to reach the network:

1. an explicit path passed to :func:`langaai.load` or :func:`load_state`;
2. the ``LANGAAI_CHECKPOINT`` environment variable;
3. a checkpoint already on disk, in any of :func:`search_dirs`;
4. a fresh download from :data:`CHECKPOINT_URL` into :func:`download_dir`.

Contents:
    CHECKPOINT_URL: Where the weights are fetched from.
    CHECKPOINT_FILENAME: What the file is called on disk.
    CHECKPOINT_SHA256: Expected hash, verified after every download.
    ENCODER: The ESM-2 model id the checkpoint was built on. Both the
        antibody tower and the antigen embedding step must use this exact
        id -- a different ESM-2 size produces embeddings of the wrong width,
        and a same-width variant produces silently wrong numbers.
    DIM: The encoder's hidden width, and therefore the model's.
    search_dirs / download_dir: Where checkpoints are looked for and saved.
    download: Fetch the weights, verify them, and store them.
    resolve: Locate the checkpoint file, downloading it if necessary.
    load_state: Load the payload (weights plus the model hyperparameters
        that produced them).
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

__all__ = [
    "CHECKPOINT_URL", "CHECKPOINT_FILENAME", "CHECKPOINT_SHA256",
    "ENCODER", "DIM",
    "search_dirs", "download_dir", "download", "resolve", "load_state",
]

#: Where the weights live. Any URL ``urllib`` can open works; override with
#: the ``LANGAAI_CHECKPOINT_URL`` environment variable.
CHECKPOINT_URL = os.environ.get(
    "LANGAAI_CHECKPOINT_URL",
    "https://huggingface.co/faisalashraf/langaai/resolve/main/langaai.pt",
)

CHECKPOINT_FILENAME = "langaai.pt"

#: SHA-256 of the released checkpoint. A download that does not match is
#: discarded rather than handed to ``torch.load``, because a truncated file
#: otherwise surfaces as an unintelligible unpickling error. Set
#: ``LANGAAI_SKIP_HASH_CHECK=1`` when serving your own weights.
CHECKPOINT_SHA256 = "7ddebbc3767b4d71862aaed0b2d7ee92efed938488b9c4a8ade23e2a67a827f7"

ENCODER = "facebook/esm2_t12_35M_UR50D"
DIM = 480

#: ``transformers`` reorganised ESM's rotary-embedding parameters in 5.0:
#: they moved from one tensor per attention layer to a single tensor on the
#: model. This checkpoint stores the 5.x layout, so loading it under 4.x
#: fails with a wall of missing/unexpected keys. Checked eagerly, because
#: that error message does not explain itself.
MIN_TRANSFORMERS = (5, 0)

_PACKAGE_DIR = Path(__file__).resolve().parent


def search_dirs() -> List[Path]:
    """Directories checked for an existing checkpoint, in priority order.

    ``LANGAAI_CHECKPOINT_DIR`` comes first when set, then the package's own
    ``checkpoints`` directory (where downloads land), then a ``checkpoints``
    directory beside the package, which is the layout of a source clone.
    """
    dirs = []
    from_env = os.environ.get("LANGAAI_CHECKPOINT_DIR")
    if from_env:
        dirs.append(Path(from_env).expanduser())
    dirs.append(_PACKAGE_DIR / "checkpoints")
    dirs.append(_PACKAGE_DIR.parent / "checkpoints")
    return dirs


def _is_writable(directory: Path) -> bool:
    """Whether we could create ``directory`` and write a file into it."""
    probe = directory
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return os.access(probe, os.W_OK)


def download_dir() -> Path:
    """Where a downloaded checkpoint is saved.

    The first writable entry of :func:`search_dirs`, falling back to
    ``~/.cache/langaai`` when the package lives somewhere read-only (a
    system-wide install, or a container image).
    """
    for directory in search_dirs():
        if _is_writable(directory):
            return directory
    return Path.home() / ".cache" / "langaai"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report(done: int, total: int) -> None:
    """One-line download progress on stderr, only when it is a terminal."""
    if not sys.stderr.isatty():
        return
    if total > 0:
        pct = 100 * done / total
        bar = "#" * int(pct // 2.5)
        sys.stderr.write(
            f"\r  downloading langaai.pt  [{bar:<40}] "
            f"{pct:5.1f}%  {done / 1e6:.0f}/{total / 1e6:.0f} MB"
        )
    else:
        sys.stderr.write(f"\r  downloading langaai.pt  {done / 1e6:.0f} MB")
    sys.stderr.flush()


def download(
    dest_dir: Optional[Path] = None, url: Optional[str] = None, force: bool = False,
) -> Path:
    """Fetch the weights and store them, returning the saved path.

    The file is streamed to a temporary name in the destination directory,
    checked against :data:`CHECKPOINT_SHA256`, and only then moved into
    place, so an interrupted download can never leave a half-written
    checkpoint behind.

    Args:
        dest_dir: Where to save. Defaults to :func:`download_dir`.
        url: Where to fetch from. Defaults to :data:`CHECKPOINT_URL`.
        force: Re-download even if the file is already there.

    Returns:
        The path to the saved checkpoint.

    Raises:
        RuntimeError: If the download fails, or its hash does not match.
    """
    directory = Path(dest_dir) if dest_dir else download_dir()
    target = directory / CHECKPOINT_FILENAME
    if target.exists() and not force:
        return target

    source = url or CHECKPOINT_URL
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot create {directory} to save the weights: {exc}") from exc

    print(f"Downloading LangAAI weights from {source}", file=sys.stderr)
    print(f"  to {target}  (about 195 MB, one time)", file=sys.stderr)

    handle = None
    try:
        with urllib.request.urlopen(source) as response:
            total = int(response.headers.get("Content-Length") or 0)
            handle = tempfile.NamedTemporaryFile(
                dir=directory, prefix=".langaai-", suffix=".part", delete=False
            )
            done = 0
            with handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    _report(done, total)
        if sys.stderr.isatty():
            sys.stderr.write("\n")

        partial = Path(handle.name)
        if total and partial.stat().st_size != total:
            raise RuntimeError(
                f"truncated download: got {partial.stat().st_size} bytes, "
                f"expected {total}"
            )

        if CHECKPOINT_SHA256 and not os.environ.get("LANGAAI_SKIP_HASH_CHECK"):
            actual = _sha256(partial)
            if actual != CHECKPOINT_SHA256:
                raise RuntimeError(
                    f"checksum mismatch: expected {CHECKPOINT_SHA256}, got "
                    f"{actual}. The file at {source} is not the checkpoint "
                    "this version of langaai expects. If you are serving your "
                    "own weights, set LANGAAI_SKIP_HASH_CHECK=1."
                )

        # Atomic within a filesystem, so readers never see a partial file.
        os.replace(partial, target)
        print(f"Saved {target}", file=sys.stderr)
        return target

    except (urllib.error.URLError, OSError, RuntimeError) as exc:
        if handle is not None:
            Path(handle.name).unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the weights from {source} ({exc}).\n"
            f"Download the file yourself and put it at {target}, or point "
            "LANGAAI_CHECKPOINT at it, or pass "
            "langaai.load(checkpoint_path=...)."
        ) from exc


def _check_transformers_version() -> None:
    import transformers

    version = transformers.__version__
    parts = []
    for piece in version.split(".")[:2]:
        digits = "".join(c for c in piece if c.isdigit())
        if not digits:
            return  # unparseable (a dev build); assume the user knows
        parts.append(int(digits))

    if tuple(parts) < MIN_TRANSFORMERS:
        raise ImportError(
            f"LangAAI needs transformers >= {'.'.join(map(str, MIN_TRANSFORMERS))}, "
            f"found {version}. The checkpoint stores ESM-2's rotary embeddings in "
            "the layout transformers 5.0 introduced; under 4.x, loading it fails "
            "with missing keys for every layer's 'rotary_embeddings.inv_freq'. "
            "Run: pip install --upgrade 'transformers>=5.0'"
        )


def resolve(checkpoint_path: Optional[Path] = None, allow_download: bool = True) -> Path:
    """Locate the checkpoint file, downloading it if necessary.

    Args:
        checkpoint_path: An explicit path, used as-is; it must exist.
            ``None`` searches the environment, then disk, then the network.
        allow_download: Set ``False`` to fail instead of reaching the
            network when no local copy is found.

    Returns:
        A path to an existing checkpoint file.

    Raises:
        FileNotFoundError: If an explicit path does not exist, or nothing
            was found and ``allow_download`` is ``False``.
        RuntimeError: If a download was attempted and failed.
    """
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}")
        return path

    from_env = os.environ.get("LANGAAI_CHECKPOINT")
    if from_env:
        path = Path(from_env).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"LANGAAI_CHECKPOINT points at {path}, which does not exist."
            )
        return path

    for directory in search_dirs():
        candidate = directory / CHECKPOINT_FILENAME
        if candidate.exists():
            return candidate

    if not allow_download:
        looked = "\n".join(f"  {d / CHECKPOINT_FILENAME}" for d in search_dirs())
        raise FileNotFoundError(
            f"No checkpoint found and downloading is disabled. Looked in:\n{looked}"
        )
    return download()


def load_state(checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the checkpoint payload, downloading the weights if needed.

    Args:
        checkpoint_path: Passed through to :func:`resolve`.

    Returns:
        A dict with ``model_state_dict`` (the weights) and ``config``, whose
        ``"model"`` entry holds the :class:`~langaai.model.ModelConfig`
        keyword arguments that produced them.

    Raises:
        ImportError: If the installed ``transformers`` is too old to load
            this checkpoint's state-dict layout.
        FileNotFoundError: If no checkpoint could be found.
    """
    _check_transformers_version()
    path = resolve(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    if "model_state_dict" not in payload or "config" not in payload:
        raise ValueError(
            f"{path} is not a LangAAI checkpoint: expected 'model_state_dict' "
            f"and 'config' keys, found {sorted(payload)}."
        )
    return payload
