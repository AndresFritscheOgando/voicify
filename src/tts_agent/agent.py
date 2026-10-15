"""TTSAgent: validate the input, synthesize it chunk by chunk, and write the audio file."""

from __future__ import annotations

import dataclasses
import difflib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tts_agent.config import (
    CHUNK_ERROR_POLICIES,
    DEFAULT_VOICE_ALIAS,
    Settings,
    validate_speed,
)
from tts_agent.engines.base import TTSEngine, VoiceInfo, WordTiming
from tts_agent.errors import (
    EmptyTextError,
    InvalidOptionError,
    ModelLoadError,
    SynthesisError,
    UnsupportedVoiceError,
)
from tts_agent.processing.audio_processor import (
    OUTPUT_FORMATS,
    concatenate,
    duration_seconds,
    fade,
    leading_silence,
    silence,
    trailing_silence,
    write_audio,
    write_wav,
)
from tts_agent.processing.normalizer import NORMALIZATION_RULES
from tts_agent.processing.pronunciations import load_pronunciations
from tts_agent.processing.spoken import Rule, map_words, to_spoken
from tts_agent.processing.subtitles import build_cues, write_srt
from tts_agent.processing.text_processor import Chunk, chunk_text
from tts_agent.utils.file_utils import read_text_file, resolve_output_path

logger = logging.getLogger(__name__)

DEFAULT_FILENAME = "narration"
DEFAULT_FORMAT = "wav"


@dataclass(frozen=True)
class ChunkTiming:
    """Where one synthesized chunk sits in the output audio, in seconds.

    `words` are the engine's word timings shifted to the same clock (empty if it has none).
    """

    index: int
    text: str
    start: float
    end: float
    words: tuple[WordTiming, ...] = ()


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
    timeline: list[ChunkTiming] = field(default_factory=list)
    load_seconds: float = 0.0  # engine/model loading before synthesis; 0 once loaded
    wav_path: Path | None = None  # the WAV kept next to an MP3 (keep_wav=True)


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
        text: str | None = None,
        *,
        file: str | Path | None = None,
        voice: str | None = DEFAULT_VOICE_ALIAS,
        speed: float | None = None,
        output_path: str | Path | None = None,
        format: str | None = None,
        overwrite: bool = False,
        on_chunk_error: str | None = None,
        subtitles: bool | None = None,
        keep_wav: bool = False,
        pronunciations: str | Path | None = None,
    ) -> GenerationResult:
        """Synthesize `text` (or the UTF-8 `.txt` script at `file`) and write an audio file.

        Exactly one of `text` and `file` must be given. The text is split into sentence-aligned
        chunks that are synthesized in order with the same voice and speed, joined with short
        pauses (longer at paragraph breaks).

        `voice="default"` (or None) uses `TTS_DEFAULT_VOICE`; `speed=None` uses
        `TTS_DEFAULT_SPEED` and must be within 0.5-2.0. `output_path` defaults to
        `<output_dir>/narration.<format>`. `format` defaults to the extension of `output_path`,
        or `"wav"`; when both are given they must agree. An existing file gets a numeric suffix
        unless `overwrite`; the suffix is chosen so the subtitle and kept WAV names are free too.

        `subtitles` (default `TTS_SUBTITLES`) writes `<output stem>.srt` next to the audio.
        `keep_wav=True` with MP3 output also writes `<output stem>.wav`.

        Script controls: `[pause]`, `[pause 1.5s]` and `[pause 800ms]` insert exact silences.
        `pronunciations` (default `TTS_PRONUNCIATIONS`) is a dictionary file of respellings or
        phonemes; it and the number/abbreviation fixes (`TTS_NORMALIZE`) change only what the
        engine reads, never the subtitles.

        A chunk that still fails after `TTS_CHUNK_RETRIES` retries is skipped and listed in
        `failed_chunks` (`on_chunk_error="skip"`), or stops the run (`"abort"`). If every chunk
        fails, `SynthesisError` is raised and no file is written.
        """
        text = self._read_input(text, file)
        voice = self.resolve_voice(voice)
        speed = validate_speed(self.settings.default_speed if speed is None else speed)
        policy = (on_chunk_error or self.settings.on_chunk_error).strip().lower()
        if policy not in CHUNK_ERROR_POLICIES:
            raise InvalidOptionError(
                f"on_chunk_error must be one of {', '.join(CHUNK_ERROR_POLICIES)}, got {policy!r}"
            )
        target, output_format = self._target_path(output_path, format)
        if keep_wav and output_format != "mp3":
            raise InvalidOptionError("keep_wav only applies to MP3 output")
        subtitles = self.settings.subtitles if subtitles is None else subtitles
        companions = ([".srt"] if subtitles else []) + ([".wav"] if keep_wav else [])
        rules = self._spoken_rules(pronunciations)
        chunks = chunk_text(text, self.settings.max_chunk_chars)

        logger.info(
            "generate_start engine=%s voice=%s speed=%s chars=%d chunks=%d",
            self.engine.name,
            voice,
            speed,
            len(text),
            len(chunks),
        )
        loading = time.perf_counter()
        self.engine.prepare(voice)  # one-time model load, kept out of generation_seconds
        load_seconds = time.perf_counter() - loading

        started = time.perf_counter()
        audio, timeline, failed = self._synthesize_chunks(chunks, voice, speed, policy, rules)
        generation_seconds = time.perf_counter() - started

        rate = self.engine.sample_rate
        path = resolve_output_path(target, overwrite, companions=companions)
        write_audio(audio, rate, path, output_format)
        wav_path = write_wav(audio, rate, path.with_suffix(".wav")) if keep_wav else None
        duration = duration_seconds(audio, rate)
        subtitle_path = None
        if subtitles:
            cues = build_cues(timeline, duration)
            subtitle_path = write_srt(cues, path.with_suffix(".srt"))
        log = logger.warning if failed else logger.info
        log(
            "generate_done path=%s duration_s=%.2f generation_s=%.2f rtf=%.3f chunks=%d failed=%s",
            path,
            duration,
            generation_seconds,
            generation_seconds / duration if duration else 0.0,
            len(chunks),
            failed or "none",
        )
        return GenerationResult(
            output_path=path,
            duration_seconds=duration,
            sample_rate=self.engine.sample_rate,
            voice=voice,
            generation_seconds=generation_seconds,
            chunk_count=len(chunks),
            failed_chunks=failed,
            subtitle_path=subtitle_path,
            format=output_format,
            timeline=timeline,
            load_seconds=load_seconds,
            wav_path=wav_path,
        )

    def _spoken_rules(self, pronunciations: str | Path | None) -> list[Rule]:
        """Pronunciation entries first (they win), then the normalization rules."""
        path = pronunciations if pronunciations is not None else self.settings.pronunciations
        rules = load_pronunciations(path) if path else []
        if self.settings.normalize:
            rules.extend(NORMALIZATION_RULES)
        return rules

    @staticmethod
    def _read_input(text: str | None, file: str | Path | None) -> str:
        if text is not None and file is not None:
            raise InvalidOptionError("Pass either text or file, not both")
        if file is not None:
            return read_text_file(file)
        if text is None or not text.strip():
            raise EmptyTextError("Text is empty; nothing to synthesize")
        return text

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
        message += " Run --list-voices (or call list_voices()) to see all voices."
        raise UnsupportedVoiceError(message)

    def _synthesize_chunks(
        self,
        chunks: list[Chunk],
        voice: str,
        speed: float,
        policy: str,
        rules: Sequence[Rule],
    ) -> tuple[np.ndarray, list[ChunkTiming], list[int]]:
        """Synthesize every chunk in order and join them with pauses.

        Between chunks goes the sentence pause, or the paragraph pause after a paragraph break.
        A `[pause]` marker replaces either with an exact silence: the engine's own leading and
        trailing silence next to the marker is trimmed, so the audible gap is the requested one.
        Segments are collected in a list and concatenated once at the end.
        """
        rate = self.engine.sample_rate
        segments: list[np.ndarray] = []
        timeline: list[ChunkTiming] = []
        failed: list[int] = []
        position = 0  # samples so far
        paragraph_break_pending = False
        marker_ms: int | None = chunks[0].pause_before_ms if chunks else None
        total = len(chunks)

        for chunk in chunks:
            spoken = to_spoken(chunk.text, rules)
            if spoken.edits:
                logger.debug("spoken_text chunk=%d text=%r", chunk.index + 1, spoken.text)
            synthesized = self._synthesize_with_retries(chunk, spoken.text, voice, speed, total)
            if synthesized is None:
                failed.append(chunk.index)
                if policy == "abort":
                    raise SynthesisError(
                        f"Chunk {chunk.index + 1}/{total} failed; aborting (on_chunk_error=abort)"
                    )
            else:
                audio, words = synthesized
                head = 0
                if marker_ms is not None:
                    head = leading_silence(audio, rate)
                    audio = fade(audio[head:], rate, fade_in=True)
                    if timeline:
                        position -= self._trim_tail(segments, timeline, rate)
                    gap = silence(marker_ms, rate)
                elif timeline:
                    pause = self.settings.paragraph_pause_ms if paragraph_break_pending else None
                    gap = silence(pause or self.settings.sentence_pause_ms, rate)
                else:
                    gap = silence(0, rate)
                segments.append(gap)
                position += len(gap)
                marker_ms = None
                paragraph_break_pending = False

                offset = position / rate
                shift = offset - head / rate
                end = (position + len(audio)) / rate
                timeline.append(
                    ChunkTiming(
                        chunk.index,
                        chunk.text,
                        offset,
                        end,
                        tuple(
                            WordTiming(
                                w.text,
                                min(max(w.start + shift, offset), end),
                                min(max(w.end + shift, offset), end),
                            )
                            for w in map_words(words, spoken)
                        ),
                    )
                )
                segments.append(audio)
                position += len(audio)
            paragraph_break_pending = paragraph_break_pending or chunk.ends_paragraph
            if chunk.pause_after_ms is not None:
                marker_ms = (marker_ms or 0) + chunk.pause_after_ms
            done = chunk.index + 1
            logger.info("chunk %d/%d (%d%%)", done, total, done * 100 // total)

        if not timeline:
            raise SynthesisError(f"All {total} chunk(s) failed to synthesize; no file was written")
        if marker_ms is not None:  # a [pause] after the last spoken chunk
            position -= self._trim_tail(segments, timeline, rate)
            segments.append(silence(marker_ms, rate))
        return concatenate(segments), timeline, failed

    @staticmethod
    def _trim_tail(segments: list[np.ndarray], timeline: list[ChunkTiming], rate: int) -> int:
        """Cut the trailing silence of the last spoken chunk; returns the samples removed."""
        audio = segments[-1]
        tail = trailing_silence(audio, rate)
        if tail == 0:
            return 0
        segments[-1] = fade(audio[: len(audio) - tail], rate, fade_in=False)
        last = timeline[-1]
        end = last.end - tail / rate
        words = tuple(WordTiming(w.text, min(w.start, end), min(w.end, end)) for w in last.words)
        timeline[-1] = dataclasses.replace(last, end=end, words=words)
        return tail

    def _synthesize_with_retries(
        self, chunk: Chunk, text: str, voice: str, speed: float, total: int
    ) -> tuple[np.ndarray, list[WordTiming]] | None:
        attempts = 1 + self.settings.chunk_retries
        for attempt in range(1, attempts + 1):
            try:
                audio, words = self.engine.synthesize_timed(text, voice, speed)
                if audio is None or len(audio) == 0:
                    raise SynthesisError("the engine returned no audio")
                return np.asarray(audio, dtype=np.float32).reshape(-1), list(words)
            except ModelLoadError:
                raise
            except Exception as exc:  # any engine failure is retried, then skipped or aborted
                logger.warning(
                    "chunk_failed chunk=%d/%d attempt=%d/%d error=%r text=%r",
                    chunk.index + 1,
                    total,
                    attempt,
                    attempts,
                    str(exc),
                    chunk.text[:60],
                )
        return None

    def _target_path(self, output_path: str | Path | None, format: str | None) -> tuple[Path, str]:
        """Resolve the output path and format, checking that they agree."""
        supported = ", ".join(OUTPUT_FORMATS)
        requested = format.strip().lower().lstrip(".") if format else None
        if requested is not None and requested not in OUTPUT_FORMATS:
            raise InvalidOptionError(f"Unsupported format {format!r}; supported: {supported}")

        if output_path is None:
            fmt = requested or DEFAULT_FORMAT
            return self.settings.output_dir / f"{DEFAULT_FILENAME}.{fmt}", fmt
        path = Path(output_path)
        extension = path.suffix.lower().lstrip(".")
        if not extension:
            fmt = requested or DEFAULT_FORMAT
            return path.with_name(f"{path.name}.{fmt}"), fmt
        if extension not in OUTPUT_FORMATS:
            raise InvalidOptionError(
                f"Unsupported output extension {path.suffix!r}; supported: {supported}"
            )
        if requested is not None and requested != extension:
            raise InvalidOptionError(
                f"Output path {path.name!r} does not match format {requested!r}; "
                f"use a .{requested} file name or drop the format"
            )
        return path, extension
