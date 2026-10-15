"""TTSAgent: validate the input, synthesize it, and write the audio file."""

from __future__ import annotations

import difflib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from tts_agent.config import DEFAULT_VOICE_ALIAS, Settings, validate_speed
from tts_agent.engines.base import TTSEngine, VoiceInfo
from tts_agent.errors import (
    EmptyTextError,
    InvalidOptionError,
    SynthesisError,
    UnsupportedVoiceError,
)
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
        voice: str | None = DEFAULT_VOICE_ALIAS,
        output_path: str | Path | None = None,
        speed: float | None = None,
        overwrite: bool = False,
    ) -> GenerationResult:
        """Synthesize `text` and write it as a WAV file.

        `voice="default"` (or None) uses `TTS_DEFAULT_VOICE`; `speed=None` uses
        `TTS_DEFAULT_SPEED` and must be within 0.5-2.0. `output_path` defaults to
        `<output_dir>/narration.wav`. An existing file gets a numeric suffix unless `overwrite`.
        """
        if text is None or not text.strip():
            raise EmptyTextError("Text is empty; nothing to synthesize")
        text = text.strip()
        voice = self.resolve_voice(voice)
        speed = validate_speed(self.settings.default_speed if speed is None else speed)
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

    def resolve_voice(self, voice: str | None = DEFAULT_VOICE_ALIAS) -> str:
        """Map `"default"`/None to the configured default and check the id against the catalog."""
        requested = (voice or DEFAULT_VOICE_ALIAS).strip().lower()
        from_default = requested == DEFAULT_VOICE_ALIAS
        if from_default:
            requested = self.settings.default_voice.strip().lower()
        known = [v.id for v in self.engine.list_voices()]
        if requested in known:
            return requested

        message = f"Unsupported voice {requested!r}"
        if from_default:
            message += " (from TTS_DEFAULT_VOICE)"
        suggestions = difflib.get_close_matches(requested, known, n=3, cutoff=0.6)
        if suggestions:
            message += f". Did you mean: {', '.join(suggestions)}?"
        message += " Call list_voices() to see all voices."
        raise UnsupportedVoiceError(message)

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
