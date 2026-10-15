"""A deterministic engine for tests. It never loads a model."""

from __future__ import annotations

from typing import Any

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


class FailingMockEngine(MockEngine):
    """A `MockEngine` that raises for chosen texts.

    A text containing any of `fail_on` fails `failures` times (forever if None) before succeeding.
    Use `fail_on=("",)` to make every text fail.
    """

    def __init__(
        self, fail_on: tuple[str, ...] = (), failures: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.fail_on = fail_on
        self.failures = failures
        self.attempts: dict[str, int] = {}

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        if any(marker in text for marker in self.fail_on):
            count = self.attempts[text] = self.attempts.get(text, 0) + 1
            if self.failures is None or count <= self.failures:
                self.calls.append((text, voice, speed))
                raise RuntimeError(f"mock failure #{count}")
        return super().synthesize(text, voice, speed)
