"""TTSAgent: validate the input, synthesize it, and write the audio file."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from tts_agent.config import Settings
from tts_agent.engines.base import TTSEngine, VoiceInfo
from tts_agent.errors import EmptyTextError, InvalidOptionError, SynthesisError
from tts_agent.processing.audio_processor import duration_seconds, write_wav
from tts_agent.utils.file_utils import resolve_output_path

logger = logging.getLogger(__name__)

DEFAULT_FILENAME = "narration"


@dataclass
class GenerationResult:
    output_path: Path
    duration_seconds: float
    sample_rate: int
    voice: str
    generation_seconds: float
    chunk_count: int
    failed_chunks: list[int] = field(default_factory=list)
    subtitle_path: Path | None = None
    format: str = "wav"


class TTSAgent:
    """Turns text into an audio file.

    `engine` defaults to Kokoro on the device from `settings`; pass any `TTSEngine` to replace it.
    """

    def __init__(self, engine: TTSEngine | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else Settings.from_env()
        if engine is None:
            from tts_agent.engines.kokoro_engine import KokoroEngine

            engine = KokoroEngine(device=self.settings.device)
        self.engine = engine

    def list_voices(self) -> list[VoiceInfo]:
        return self.engine.list_voices()

    def generate(
        self,
        text: str,
        voice: str | None = None,
        output_path: str | Path | None = None,
        speed: float | None = None,
        overwrite: bool = False,
    ) -> GenerationResult:
        """Synthesize `text` and write it as a WAV file.

        `voice` and `speed` default to the configured defaults. `output_path` defaults to
        `<output_dir>/narration.wav`. An existing file gets a numeric suffix unless `overwrite`.
        """
        if text is None or not text.strip():
            raise EmptyTextError("Text is empty; nothing to synthesize")
        text = text.strip()
        voice = voice or self.settings.default_voice
        speed = self.settings.default_speed if speed is None else speed
        target = self._target_path(output_path)

        logger.info(
            "generate_start engine=%s voice=%s speed=%s chars=%d",
            self.engine.name,
            voice,
            speed,
            len(text),
        )
        started = time.perf_counter()
        audio = self.engine.synthesize(text, voice, speed)
        if audio is None or len(audio) == 0:
            raise SynthesisError("The engine returned no audio")
        generation_seconds = time.perf_counter() - started

        path = write_wav(audio, self.engine.sample_rate, resolve_output_path(target, overwrite))
        duration = duration_seconds(audio, self.engine.sample_rate)
        logger.info(
            "generate_done path=%s duration_s=%.2f generation_s=%.2f rtf=%.3f",
            path,
            duration,
            generation_seconds,
            generation_seconds / duration if duration else 0.0,
        )
        return GenerationResult(
            output_path=path,
            duration_seconds=duration,
            sample_rate=self.engine.sample_rate,
            voice=voice,
            generation_seconds=generation_seconds,
            chunk_count=1,
        )

    def _target_path(self, output_path: str | Path | None) -> Path:
        if output_path is None:
            return self.settings.output_dir / f"{DEFAULT_FILENAME}.wav"
        path = Path(output_path)
        if path.suffix == "":
            return path.with_suffix(".wav")
        if path.suffix.lower() != ".wav":
            raise InvalidOptionError(
                f"Unsupported output extension {path.suffix!r}; only .wav is supported"
            )
        return path
