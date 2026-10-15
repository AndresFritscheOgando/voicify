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


# Every English voice file in hexgrad/Kokoro-82M (v1.0), as listed in its VOICES.md.
ENGLISH_VOICE_IDS = {
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
    "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx",
    "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
}  # fmt: skip


def test_construction_does_not_load_the_model():
    engine = KokoroEngine()

    assert engine.device is None
    assert engine.sample_rate == 24_000
    assert engine.name == "kokoro"


def test_catalog_is_the_english_voice_set():
    voices = KokoroEngine().list_voices()

    assert {v.id for v in voices} == ENGLISH_VOICE_IDS
    assert len(voices) == len(ENGLISH_VOICE_IDS)


def test_catalog_metadata_matches_voice_ids():
    for v in KokoroEngine().list_voices():
        assert v.language == {"a": "en-us", "b": "en-gb"}[v.id[0]]
        assert v.accent == {"a": "American", "b": "British"}[v.id[0]]
        assert v.gender == {"f": "female", "m": "male"}[v.id[1]]
        assert v.name == v.id[3:].capitalize()
        assert v.description.startswith("Grade ")


def test_list_voices_returns_a_copy():
    engine = KokoroEngine()
    engine.list_voices().clear()

    assert len(engine.list_voices()) == len(ENGLISH_VOICE_IDS)


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


@pytest.mark.model
def test_real_kokoro_speed_changes_duration():
    engine = KokoroEngine()
    text = "The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun."

    slow = len(engine.synthesize(text, "af_heart", speed=0.75))
    normal = len(engine.synthesize(text, "af_heart", speed=1.0))
    fast = len(engine.synthesize(text, "af_heart", speed=1.5))

    assert slow > normal > fast
    # Kokoro scales predicted durations by 1/speed, so the ratio should be close to 2.
    assert slow / fast == pytest.approx(2.0, rel=0.2)


@pytest.mark.model
def test_real_kokoro_voices_differ():
    engine = KokoroEngine()
    text = "Welcome to our channel."

    clips = {v: engine.synthesize(text, v) for v in ("af_heart", "am_michael", "bf_emma")}

    for a in clips:
        for b in clips:
            if a < b:
                n = min(len(clips[a]), len(clips[b]))
                assert not np.allclose(clips[a][:n], clips[b][:n], atol=1e-3), (a, b)
