"""The engine abstraction every TTS backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VoiceInfo:
    """A voice an engine can speak with."""

    id: str  # "af_heart"
    name: str  # "Heart"
    language: str  # "en-us" / "en-gb"
    gender: str  # "female" / "male"
    accent: str = ""  # "American" / "British"
    description: str = ""  # quality note, e.g. "Grade A"


@dataclass(frozen=True)
class WordTiming:
    """When one spoken word starts and ends, in seconds."""

    text: str
    start: float
    end: float


class TTSEngine(ABC):
    """A text-to-speech backend.

    `synthesize` returns float32 mono audio in [-1, 1] at `sample_rate`.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @abstractmethod
    def list_voices(self) -> list[VoiceInfo]: ...

    @abstractmethod
    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray: ...

    def synthesize_timed(
        self, text: str, voice: str, speed: float = 1.0
    ) -> tuple[np.ndarray, list[WordTiming]]:
        """Audio plus per-word timings relative to the start of that audio.

        Engines that cannot time words return an empty list; subtitles then fall back to
        splitting time by character count.
        """
        return self.synthesize(text, voice, speed), []

    @property
    def default_voice(self) -> str | None:
        """The voice `"default"` means for this engine, or None to use `TTS_DEFAULT_VOICE`."""
        return None

    @property
    def supports_phonemes(self) -> bool:
        """Whether the engine reads misaki's `[word](/phonemes/)` syntax in its input."""
        return True

    @property
    def max_chunk_chars(self) -> int | None:
        """The longest chunk the engine handles well, or None for no limit beyond the settings."""
        return None

    def plausible_seconds(self, text: str, speed: float = 1.0) -> tuple[float, float] | None:
        """The range of audio lengths that is believable for `text` at `speed`, or None to accept
        any length. A take outside it counts as a failed attempt and is synthesized again."""
        return None

    @property
    def cache_key(self) -> str:
        """Identifies everything besides voice, speed and text that shapes the audio, so saved
        chunks are only reused with the same engine settings. Defaults to `name`."""
        return self.name

    def prepare(self, voice: str) -> None:  # noqa: B027 - optional hook, a no-op by default
        """Load whatever `voice` needs before the first `synthesize` call. Optional."""
