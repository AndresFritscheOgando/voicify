"""File and path helpers."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from tts_agent.errors import EmptyTextError, InputFileError, OutputError

INPUT_EXTENSIONS = (".txt",)


def read_text_file(path: str | Path) -> str:
    """Read a UTF-8 `.txt` script (a byte-order mark is allowed).

    Raises `InputFileError` if the file is missing, not a `.txt` file, unreadable, or not UTF-8,
    and `EmptyTextError` if it holds only whitespace.
    """
    path = Path(path)
    if not path.exists():
        raise InputFileError(f"Input file not found: {path}")
    if not path.is_file():
        raise InputFileError(f"Input path is not a file: {path}")
    if path.suffix.lower() not in INPUT_EXTENSIONS:
        raise InputFileError(f"Input file must be a .txt file, got {path.name!r}")
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputFileError(f"Input file {path} is not valid UTF-8: {exc.reason}") from exc
    except OSError as exc:
        raise InputFileError(f"Could not read input file {path}: {exc.strerror or exc}") from exc
    if not text.strip():
        raise EmptyTextError(f"Input file {path} is empty")
    return text


def atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    """Call `write` on a temporary file next to `path`, then rename it over `path`.

    A write that fails or is interrupted leaves an existing file at `path` untouched, and never
    leaves a half-written one behind.
    """
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise


def resolve_output_path(
    path: str | Path, overwrite: bool = False, companions: Sequence[str] = ()
) -> Path:
    """Return the path to write to, creating missing parent directories.

    If the file already exists and `overwrite` is False, a numeric suffix is added:
    `narration.wav` -> `narration_1.wav` -> `narration_2.wav`. `companions` are extensions of
    files written next to it (such as `.srt`); a name is only taken if those are free too.
    """
    path = Path(path)
    if path.is_dir():
        raise OutputError(f"Output path {path} is a directory, not a file")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputError(f"Could not create output directory {path.parent}: {exc}") from exc

    def taken(candidate: Path) -> bool:
        return candidate.exists() or any(candidate.with_suffix(c).exists() for c in companions)

    if overwrite or not taken(path):
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not taken(candidate):
            return candidate
        counter += 1
