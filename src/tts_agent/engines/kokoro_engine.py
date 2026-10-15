"""Kokoro-82M engine. The model loads lazily on first use and is reused afterwards."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from tts_agent.config import DEVICES
from tts_agent.engines.base import TTSEngine, VoiceInfo
from tts_agent.errors import InvalidOptionError, SynthesisError, UnsupportedVoiceError
from tts_agent.logging_setup import silence_model_warnings

if TYPE_CHECKING:
    from kokoro import KModel, KPipeline

logger = logging.getLogger(__name__)

REPO_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24_000

# Kokoro picks its G2P from the first letter of the voice id: a = American, b = British.
_LANG_CODES = {"a": "a", "b": "b"}

# The full English catalog arrives in Phase 4. Until then only the default voice is listed.
_VOICES = [
    VoiceInfo("af_heart", "Heart", "en-us", "female", "Default voice"),
]


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
        return list(_VOICES)

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        pipeline = self._pipeline_for(voice)
        try:
            segments = [
                _to_numpy(result.audio)
                for result in pipeline(text, voice=voice, speed=speed)
                if result.audio is not None
            ]
        except Exception as exc:
            raise SynthesisError(f"Kokoro failed with voice {voice!r}: {exc}") from exc
        if not segments:
            raise SynthesisError("Kokoro produced no audio for the given text")
        return np.concatenate(segments).astype(np.float32, copy=False)

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

            model = KModel(repo_id=REPO_ID).to(device).eval()
        except Exception as exc:
            raise SynthesisError(f"Failed to load the Kokoro model on {device}: {exc}") from exc
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
                raise SynthesisError(
                    "TTS_DEVICE=cuda but CUDA is not available. Use auto or cpu, "
                    "or install a CUDA build of PyTorch."
                )
            return "cuda"
        if self._requested_device == "auto" and not cuda:
            logger.warning("cuda_unavailable fallback=cpu")
            return "cpu"
        return "cuda" if self._requested_device == "auto" else "cpu"


def _to_numpy(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)
