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

    def prepare(self, voice: str) -> None:  # noqa: B027 - optional hook, a no-op by default
        """Load whatever `voice` needs before the first `synthesize` call. Optional."""
