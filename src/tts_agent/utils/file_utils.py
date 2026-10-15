"""File and path helpers."""

from __future__ import annotations

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


def resolve_output_path(path: str | Path, overwrite: bool = False) -> Path:
    """Return the path to write to, creating missing parent directories.

    If the file already exists and `overwrite` is False, a numeric suffix is added:
    `narration.wav` -> `narration_1.wav` -> `narration_2.wav`.
    """
    path = Path(path)
    if path.is_dir():
        raise OutputError(f"Output path {path} is a directory, not a file")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputError(f"Could not create output directory {path.parent}: {exc}") from exc
    if overwrite or not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
