"""A deterministic engine for tests. It never loads a model."""

from __future__ import annotations

import numpy as np

from tts_agent.engines.base import TTSEngine, VoiceInfo

MOCK_VOICES = [
    VoiceInfo("af_heart", "Heart", "en-us", "female", "American", "Mock voice"),
    VoiceInfo("am_michael", "Michael", "en-us", "male", "American", "Mock voice"),
    VoiceInfo("bf_emma", "Emma", "en-gb", "female", "British", "Mock voice"),
]


class MockEngine(TTSEngine):
    """Returns a sine wave (or silence) whose length grows with the text.

    Every call is recorded in `calls` as `(text, voice, speed)`.
    """

    def __init__(
        self,
        sample_rate: int = 24_000,
        seconds_per_char: float = 0.01,
        frequency: float = 220.0,
        silent: bool = False,
    ) -> None:
        self._sample_rate = sample_rate
        self.seconds_per_char = seconds_per_char
        self.frequency = frequency
        self.silent = silent
        self.calls: list[tuple[str, str, float]] = []

    @property
    def name(self) -> str:
        return "mock"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def list_voices(self) -> list[VoiceInfo]:
        return list(MOCK_VOICES)

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        self.calls.append((text, voice, speed))
        samples = max(1, round(len(text) * self.seconds_per_char * self._sample_rate / speed))
        if self.silent:
            return np.zeros(samples, dtype=np.float32)
        t = np.arange(samples, dtype=np.float32) / self._sample_rate
        return (0.3 * np.sin(2 * np.pi * self.frequency * t)).astype(np.float32)
