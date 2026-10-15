"""File and path helpers."""

from __future__ import annotations

from pathlib import Path

from tts_agent.errors import OutputError


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
