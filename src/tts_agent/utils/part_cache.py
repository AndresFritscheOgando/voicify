"""Finished chunks saved to disk while a long job runs, so a failed run can be resumed.

Each chunk is stored under a hash of everything that shapes its audio: the engine (and its
settings), the voice, the speed, the sample rate, and the exact text the engine reads. Editing
one sentence of a script therefore only invalidates that sentence. The folder sits next to the
output, `.<output name>.parts/`, and is removed once the audio has been written in full.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np

from tts_agent.engines.base import WordTiming

logger = logging.getLogger(__name__)

PART_SUFFIX = ".npz"


def parts_dir_for(output_path: Path) -> Path:
    """`outputs/episode.mp3` -> `outputs/.episode.mp3.parts`."""
    return output_path.with_name(f".{output_path.name}.parts")


class PartCache:
    """Content-addressed store of synthesized chunks for one output file."""

    def __init__(self, directory: Path, engine_key: str, sample_rate: int) -> None:
        self.directory = Path(directory)
        self.engine_key = engine_key
        self.sample_rate = sample_rate
        self.saved = 0  # chunks written by this run

    def key(self, text: str, voice: str, speed: float) -> str:
        identity = {
            "engine": self.engine_key,
            "rate": self.sample_rate,
            "voice": voice,
            "speed": speed,
            "text": text,
        }
        payload = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]

    def load(
        self, text: str, voice: str, speed: float
    ) -> tuple[np.ndarray, list[WordTiming]] | None:
        """The saved audio and word timings for this chunk, or None if there are none."""
        path = self._path(text, voice, speed)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                audio = np.asarray(data["audio"], dtype=np.float32).reshape(-1)
                words = [
                    WordTiming(str(t), float(s), float(e))
                    for t, s, e in zip(data["words"], data["starts"], data["ends"], strict=True)
                ]
        except Exception as exc:  # a damaged part is simply synthesized again
            logger.warning("part_unreadable path=%s error=%r", path, str(exc))
            return None
        return (audio, words) if audio.size else None

    def save(self, text: str, voice: str, speed: float, audio: np.ndarray, words) -> None:
        """Store one chunk. Written to a temporary file first, so a crash never leaves half a
        part behind. A part that cannot be saved only costs resumability, so it just warns."""
        path = self._path(text, voice, speed)
        temporary = path.with_name(f"{path.name}.tmp")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with open(temporary, "wb") as handle:
                np.savez(
                    handle,
                    audio=np.asarray(audio, dtype=np.float32),
                    words=np.array([w.text for w in words], dtype=str),
                    starts=np.array([w.start for w in words], dtype=np.float64),
                    ends=np.array([w.end for w in words], dtype=np.float64),
                )
            os.replace(temporary, path)
        except OSError as exc:
            logger.warning("part_not_saved path=%s error=%r", path, str(exc))
            with contextlib.suppress(OSError):  # e.g. NotADirectoryError when the folder is a file
                temporary.unlink(missing_ok=True)
            return
        self.saved += 1

    def count(self) -> int:
        """Saved parts on disk, from this run or earlier ones."""
        if not self.directory.is_dir():
            return 0
        return sum(1 for p in self.directory.iterdir() if p.suffix == PART_SUFFIX)

    def clear(self) -> None:
        """Remove the folder once the output no longer needs it."""
        if self.directory.is_dir():
            shutil.rmtree(self.directory, ignore_errors=True)

    def _path(self, text: str, voice: str, speed: float) -> Path:
        return self.directory / f"{self.key(text, voice, speed)}{PART_SUFFIX}"
