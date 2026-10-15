"""Kokoro-82M engine. The model loads lazily on first use and is reused afterwards."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from tts_agent.config import DEVICES
from tts_agent.engines.base import TTSEngine, VoiceInfo, WordTiming
from tts_agent.errors import (
    InvalidOptionError,
    ModelLoadError,
    SynthesisError,
    UnsupportedVoiceError,
)
from tts_agent.logging_setup import silence_model_warnings

if TYPE_CHECKING:
    from kokoro import KModel, KPipeline

logger = logging.getLogger(__name__)

REPO_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24_000

# Kokoro picks its G2P from the first letter of the voice id: a = American, b = British.
_LANG_CODES = {"a": "a", "b": "b"}

# English voices from Kokoro's VOICES.md (hexgrad/Kokoro-82M, v1.0), with its overall grade.
# The id prefix encodes accent and gender: a/b = American/British, f/m = female/male.
_GRADES = {
    "af_heart": "A",
    "af_alloy": "C",
    "af_aoede": "C+",
    "af_bella": "A-",
    "af_jessica": "D",
    "af_kore": "C+",
    "af_nicole": "B-",
    "af_nova": "C",
    "af_river": "D",
    "af_sarah": "C+",
    "af_sky": "C-",
    "am_adam": "F+",
    "am_echo": "D",
    "am_eric": "D",
    "am_fenrir": "C+",
    "am_liam": "D",
    "am_michael": "C+",
    "am_onyx": "D",
    "am_puck": "C+",
    "am_santa": "D-",
    "bf_alice": "D",
    "bf_emma": "B-",
    "bf_isabella": "C",
    "bf_lily": "D",
    "bm_daniel": "D",
    "bm_fable": "C",
    "bm_george": "C",
    "bm_lewis": "D+",
}
_ACCENTS = {"a": ("en-us", "American"), "b": ("en-gb", "British")}
_GENDERS = {"f": "female", "m": "male"}


def _voice_info(voice_id: str, grade: str) -> VoiceInfo:
    language, accent = _ACCENTS[voice_id[0]]
    return VoiceInfo(
        id=voice_id,
        name=voice_id[3:].capitalize(),
        language=language,
        gender=_GENDERS[voice_id[1]],
        accent=accent,
        description=f"Grade {grade}",
    )


VOICES = [_voice_info(voice_id, grade) for voice_id, grade in _GRADES.items()]


class KokoroEngine(TTSEngine):
    """Kokoro-82M through `kokoro.KPipeline`.

    One `KModel` is shared by a pipeline per language code (American, British).
    """

    def __init__(self, device: str = "auto") -> None:
        if device not in DEVICES:
            raise InvalidOptionError(f"device must be one of {', '.join(DEVICES)}, got {device!r}")
        self._requested_device = device
        self._device: str | None = None
        self._model: KModel | None = None
        self._pipelines: dict[str, KPipeline] = {}

    @property
    def name(self) -> str:
        return "kokoro"

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def device(self) -> str | None:
        """The device in use, or None before the model has loaded."""
        return self._device

    def list_voices(self) -> list[VoiceInfo]:
        return list(VOICES)

    def prepare(self, voice: str) -> None:
        """Import Kokoro, load the model and `voice`, and build the pipeline for its language."""
        self._voice_ready(voice)

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        return self.synthesize_timed(text, voice, speed)[0]

    def synthesize_timed(
        self, text: str, voice: str, speed: float = 1.0
    ) -> tuple[np.ndarray, list[WordTiming]]:
        """Synthesize and collect misaki's token timestamps (derived from predicted durations)."""
        pipeline = self._voice_ready(voice)
        segments: list[np.ndarray] = []
        words: list[WordTiming] = []
        offset = 0.0  # Kokoro may yield several segments; their timestamps restart at zero
        try:
            for result in pipeline(text, voice=voice, speed=speed):
                if result.audio is None:
                    continue
                audio = _to_numpy(result.audio)
                for token in result.tokens or []:
                    if token.start_ts is not None and token.end_ts is not None:
                        words.append(
                            WordTiming(token.text, offset + token.start_ts, offset + token.end_ts)
                        )
                segments.append(audio)
                offset += len(audio) / SAMPLE_RATE
        except Exception as exc:
            raise SynthesisError(f"Kokoro failed with voice {voice!r}: {exc}") from exc
        if not segments:
            raise SynthesisError("Kokoro produced no audio for the given text")
        return np.concatenate(segments).astype(np.float32, copy=False), words

    def _voice_ready(self, voice: str) -> KPipeline:
        """The pipeline for `voice`, with the voice loaded from the local cache when present."""
        pipeline = self._pipeline_for(voice)
        if voice not in pipeline.voices:
            import torch

            try:
                path = model_file(f"voices/{voice}.pt")
                pipeline.voices[voice] = torch.load(path, weights_only=True)
            except Exception as exc:
                raise SynthesisError(
                    f"Could not load Kokoro voice {voice!r}: {exc}{_offline_hint()}"
                ) from exc
        return pipeline

    def _pipeline_for(self, voice: str) -> KPipeline:
        lang_code = _LANG_CODES.get(voice[:1])
        if lang_code is None:
            raise UnsupportedVoiceError(
                f"Voice {voice!r} is not an English Kokoro voice (ids start with 'a' or 'b')"
            )
        pipeline = self._pipelines.get(lang_code)
        if pipeline is None:
            from kokoro import KPipeline

            model = self._load_model()
            pipeline = KPipeline(lang_code=lang_code, repo_id=REPO_ID, model=model)
            self._pipelines[lang_code] = pipeline
        return pipeline

    def _load_model(self) -> KModel:
        if self._model is not None:
            return self._model
        silence_model_warnings()
        device = self._resolve_device()
        started = time.perf_counter()
        try:
            from kokoro import KModel

            model = KModel(
                repo_id=REPO_ID,
                config=model_file("config.json"),
                model=model_file(KModel.MODEL_NAMES[REPO_ID]),
            )
            model = model.to(device).eval()
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load the Kokoro model on {device}: {exc}{_offline_hint()}"
            ) from exc
        self._model = model
        self._device = device
        logger.info(
            "model_loaded engine=kokoro device=%s seconds=%.2f",
            device,
            time.perf_counter() - started,
        )
        return model

    def _resolve_device(self) -> str:
        import torch

        cuda = torch.cuda.is_available()
        if self._requested_device == "cuda":
            if not cuda:
                raise ModelLoadError(
                    "TTS_DEVICE=cuda but CUDA is not available. Use auto or cpu, "
                    "or install a CUDA build of PyTorch."
                )
            return "cuda"
        if self._requested_device == "auto" and not cuda:
            logger.warning("cuda_unavailable fallback=cpu")
            return "cpu"
        return "cuda" if self._requested_device == "auto" else "cpu"


def model_file(filename: str) -> str:
    """Path to a file of the Kokoro repo: from the local Hugging Face cache when it is there,
    otherwise downloaded once.

    Kokoro would otherwise call `hf_hub_download` every time, which asks huggingface.co whether
    the file changed (three requests per run) unless HF_HUB_OFFLINE=1 is set. Resolving cached
    files ourselves means no network access at all once everything has been downloaded.
    """
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cached = try_to_load_from_cache(REPO_ID, filename)
    if isinstance(cached, str):
        return cached
    logger.info("model_download repo=%s file=%s", REPO_ID, filename)
    return hf_hub_download(REPO_ID, filename)


def _offline_hint() -> str:
    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return (
            " (HF_HUB_OFFLINE is set but this file is not cached yet: unset HF_HUB_OFFLINE and"
            " run once with a network connection to download it)"
        )
    return ""


def _to_numpy(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)
