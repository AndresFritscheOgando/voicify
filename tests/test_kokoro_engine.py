"""Kokoro engine tests. The unmarked tests never load the model; `-m model` ones do."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from tts_agent.engines.kokoro_engine import KokoroEngine
from tts_agent.errors import InvalidOptionError, SynthesisError, UnsupportedVoiceError


def test_invalid_device_is_rejected():
    with pytest.raises(InvalidOptionError):
        KokoroEngine(device="tpu")


def test_construction_does_not_load_the_model():
    engine = KokoroEngine()

    assert engine.device is None
    assert engine.sample_rate == 24_000
    assert engine.name == "kokoro"
    assert [v.id for v in engine.list_voices()] == ["af_heart"]


def test_non_english_voice_is_rejected_before_loading():
    engine = KokoroEngine()

    with pytest.raises(UnsupportedVoiceError):
        engine.synthesize("Hola", "ef_dora")
    assert engine.device is None


def test_auto_falls_back_to_cpu_with_warning(monkeypatch, caplog):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with caplog.at_level("WARNING", logger="tts_agent"):
        assert KokoroEngine(device="auto")._resolve_device() == "cpu"
    assert "cuda_unavailable" in caplog.text


def test_cuda_requested_but_unavailable_raises(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(SynthesisError, match="CUDA is not available"):
        KokoroEngine(device="cuda")._resolve_device()


def test_cpu_is_used_when_requested(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert KokoroEngine(device="cpu")._resolve_device() == "cpu"
    assert KokoroEngine(device="auto")._resolve_device() == "cuda"


@pytest.mark.model
def test_real_kokoro_generates_example_sentence(tmp_path):
    from tts_agent import Settings, TTSAgent

    agent = TTSAgent(settings=Settings(output_dir=tmp_path))
    result = agent.generate(
        "Welcome to our channel. Today we are going to explore how a small, local "
        "text-to-speech model can narrate an entire video script."
    )

    audio, rate = sf.read(result.output_path, dtype="float32")
    assert rate == 24_000
    assert 3.0 < result.duration_seconds < 20.0
    assert np.sqrt(np.mean(audio**2)) > 0.01  # audible, not silence
