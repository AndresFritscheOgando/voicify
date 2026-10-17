"""Chatterbox engine: speaks in a voice cloned from a short audio sample.

Needs the optional dependencies (`uv sync --extra clone`). The model (~3 GB) downloads from
Hugging Face on first use, loads lazily and is reused afterwards.
"""

from __future__ import annotations

import contextlib
import io
import logging
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tts_agent.config import (
    DEFAULT_CFG_WEIGHT,
    DEFAULT_EXAGGERATION,
    DEVICES,
    validate_cfg_weight,
    validate_exaggeration,
)
from tts_agent.engines.base import TTSEngine, VoiceInfo
from tts_agent.engines.common import hub_file, offline_hint, resolve_device
from tts_agent.errors import (
    InputFileError,
    InvalidOptionError,
    ModelLoadError,
    SynthesisError,
    UnsupportedVoiceError,
)

if TYPE_CHECKING:
    from chatterbox.tts import ChatterboxTTS

logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox"
MODEL_FILES = ("ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json")
SAMPLE_RATE = 24_000
CLONE_VOICE_ID = "clone"
SAMPLE_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")
MIN_SAMPLE_SECONDS = 5.0
USED_SAMPLE_SECONDS = 10.0  # Chatterbox conditions on the first 10 s of the sample
# Chatterbox stops after 1000 speech tokens (40 s) and drifts on long inputs, so chunks stay short.
MAX_CHUNK_CHARS = 300

_INSTALL_HINT = (
    "Voice cloning needs the optional Chatterbox dependencies: run `uv sync --extra clone`"
)

# Harmless warnings that Chatterbox and its dependencies print while loading and generating.
_NOISY_WARNINGS = (
    r".*pkg_resources is deprecated.*",
    r".*LoRACompatibleLinear.*",
    r".*sdp_kernel.*is deprecated.*",
)


class ChatterboxEngine(TTSEngine):
    """Chatterbox (Resemble AI, MIT) through `chatterbox.tts.ChatterboxTTS`.

    It has one voice, `clone`: the speaker of `voice_sample`, a 5+ second clip of clean speech.
    `exaggeration` (0.25-2.0) sets how expressive the delivery is; `cfg_weight` (0-1) how closely
    it follows the sample's pacing (lower is slower and more deliberate). Chatterbox cannot time
    words, so subtitles are timed by character count. Its output carries Resemble AI's inaudible
    Perth watermark, which marks the audio as AI-generated.
    """

    def __init__(
        self,
        voice_sample: str | Path,
        device: str = "auto",
        exaggeration: float = DEFAULT_EXAGGERATION,
        cfg_weight: float = DEFAULT_CFG_WEIGHT,
    ) -> None:
        if device not in DEVICES:
            raise InvalidOptionError(f"device must be one of {', '.join(DEVICES)}, got {device!r}")
        self.voice_sample = Path(voice_sample)
        self.sample_seconds = _check_sample(self.voice_sample)
        self.exaggeration = validate_exaggeration(exaggeration)
        self.cfg_weight = validate_cfg_weight(cfg_weight)
        self._requested_device = device
        self._device: str | None = None
        self._model: ChatterboxTTS | None = None

    @property
    def name(self) -> str:
        return "chatterbox"

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def device(self) -> str | None:
        """The device in use, or None before the model has loaded."""
        return self._device

    @property
    def default_voice(self) -> str:
        return CLONE_VOICE_ID

    @property
    def supports_phonemes(self) -> bool:
        return False

    @property
    def max_chunk_chars(self) -> int:
        return MAX_CHUNK_CHARS

    @property
    def cache_key(self) -> str:
        """The sample's path, size and modification time, plus the delivery settings."""
        stat = self.voice_sample.stat()
        sample = f"{self.voice_sample.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        return f"chatterbox:{sample}:{self.exaggeration}:{self.cfg_weight}"

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                id=CLONE_VOICE_ID,
                name=self.voice_sample.stem,
                language="en",
                gender="",
                description=f"Cloned from {self.voice_sample.name}",
            )
        ]

    def prepare(self, voice: str) -> None:
        """Load the model and analyse the voice sample (the first `prepare` takes a while)."""
        self._check_voice(voice)
        self._ready()

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        self._check_voice(voice)
        model = self._ready()
        try:
            with _quiet():
                wav = model.generate(
                    text, exaggeration=self.exaggeration, cfg_weight=self.cfg_weight
                )
        except Exception as exc:
            raise SynthesisError(f"Chatterbox failed: {exc}") from exc
        audio = _to_numpy(wav)
        if audio.size == 0:
            raise SynthesisError("Chatterbox produced no audio for the given text")
        if speed != 1.0:  # Chatterbox has no speed control; stretch without changing pitch
            import librosa

            audio = librosa.effects.time_stretch(audio, rate=speed)
        return audio.astype(np.float32, copy=False)

    def _check_voice(self, voice: str) -> None:
        if voice != CLONE_VOICE_ID:
            raise UnsupportedVoiceError(
                f"Voice {voice!r} is not available: Chatterbox speaks only in the cloned voice "
                f"{CLONE_VOICE_ID!r}"
            )

    def _ready(self) -> ChatterboxTTS:
        """The model, loaded on first use with the voice sample already analysed."""
        if self._model is not None:
            return self._model
        device = resolve_device(self._requested_device)
        started = time.perf_counter()
        try:
            with _quiet():
                from chatterbox.models.t3 import t3
                from chatterbox.tts import ChatterboxTTS
        except ImportError as exc:
            raise ModelLoadError(f"{_INSTALL_HINT} ({exc})") from exc
        try:
            paths = [hub_file(REPO_ID, filename) for filename in MODEL_FILES]
            with _quiet():
                _disable_progress_bar(t3)
                model = ChatterboxTTS.from_local(Path(paths[0]).parent, device)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load the Chatterbox model on {device}: {exc}{offline_hint()}"
            ) from exc
        try:
            with _quiet():
                model.prepare_conditionals(str(self.voice_sample), exaggeration=self.exaggeration)
        except Exception as exc:
            raise ModelLoadError(f"Could not use voice sample {self.voice_sample}: {exc}") from exc
        self._model = model
        self._device = device
        logger.info(
            "model_loaded engine=chatterbox device=%s seconds=%.2f sample=%s",
            device,
            time.perf_counter() - started,
            self.voice_sample,
        )
        return model


def _check_sample(path: Path) -> float:
    """Check the voice sample before any model loads; returns its length in seconds."""
    if not path.is_file():
        raise InputFileError(f"Voice sample not found: {path}")
    if path.suffix.lower() not in SAMPLE_EXTENSIONS:
        raise InputFileError(
            f"Voice sample must be one of {', '.join(SAMPLE_EXTENSIONS)}, got {path.name!r}"
        )
    import soundfile as sf

    try:
        seconds = float(sf.info(str(path)).duration)
    except Exception as exc:
        raise InputFileError(f"Could not read voice sample {path}: {exc}") from exc
    if seconds < MIN_SAMPLE_SECONDS:
        raise InvalidOptionError(
            f"Voice sample {path.name!r} is {seconds:.1f} s long; it needs at least "
            f"{MIN_SAMPLE_SECONDS:.0f} s of clean speech (about {USED_SAMPLE_SECONDS:.0f} s works "
            "best)"
        )
    if seconds > USED_SAMPLE_SECONDS + 1:
        logger.info(
            "voice_sample_trimmed seconds=%.1f used=%.0f (only the start of the sample is used)",
            seconds,
            USED_SAMPLE_SECONDS,
        )
    return seconds


@contextlib.contextmanager
def _quiet():
    """Hide the warnings, prints and transformers log lines Chatterbox emits on every call."""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        for pattern in _NOISY_WARNINGS:
            warnings.filterwarnings("ignore", message=pattern)
        transformers_logger = logging.getLogger("transformers")
        level = transformers_logger.level
        transformers_logger.setLevel(logging.ERROR)
        try:
            yield
        finally:
            transformers_logger.setLevel(level)


def _disable_progress_bar(t3_module: Any) -> None:
    """Chatterbox draws a tqdm bar for every sentence it samples; turn it off."""
    from functools import partial

    from tqdm import tqdm

    t3_module.tqdm = partial(tqdm, disable=True)


def _to_numpy(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)
