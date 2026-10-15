"""Chatterbox engine tests. The unmarked tests never load the model; `-m model` ones do."""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from tts_agent.agent import TTSAgent
from tts_agent.engines.chatterbox_engine import (
    CLONE_VOICE_ID,
    MAX_CHUNK_CHARS,
    ChatterboxEngine,
)
from tts_agent.engines.mock_engine import MockEngine
from tts_agent.errors import (
    InputFileError,
    InvalidOptionError,
    ModelLoadError,
    SynthesisError,
    UnsupportedVoiceError,
)


def write_sample(path, seconds: float = 6.0, rate: int = 24_000):
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    sf.write(path, 0.3 * np.sin(2 * np.pi * 180 * t), rate)
    return path


@pytest.fixture
def sample(tmp_path):
    return write_sample(tmp_path / "narrator.wav")


class FakeModel:
    """Stands in for `ChatterboxTTS`: one second of tone per call, recording its arguments."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, float, float]] = []
        self.fail = fail

    def generate(self, text, exaggeration, cfg_weight):
        import torch

        self.calls.append((text, exaggeration, cfg_weight))
        if self.fail:
            raise RuntimeError("out of memory")
        t = torch.arange(24_000, dtype=torch.float32) / 24_000
        return (0.3 * torch.sin(2 * torch.pi * 220 * t)).unsqueeze(0)


def loaded(engine: ChatterboxEngine, model: FakeModel) -> ChatterboxEngine:
    engine._model = model
    engine._device = "cpu"
    return engine


# --- construction and validation -------------------------------------------------------------


def test_construction_checks_the_sample_but_does_not_load_the_model(sample):
    engine = ChatterboxEngine(sample)

    assert engine.device is None
    assert engine.name == "chatterbox"
    assert engine.sample_rate == 24_000
    assert engine.sample_seconds == pytest.approx(6.0)


def test_engine_has_one_cloned_voice(sample):
    engine = ChatterboxEngine(sample)

    [voice] = engine.list_voices()
    assert voice.id == CLONE_VOICE_ID == engine.default_voice
    assert voice.name == "narrator"
    assert "narrator.wav" in voice.description


def test_engine_reads_plain_text_in_short_chunks(sample):
    engine = ChatterboxEngine(sample)

    assert engine.supports_phonemes is False
    assert engine.max_chunk_chars == MAX_CHUNK_CHARS


def test_missing_sample_is_an_input_error(tmp_path):
    with pytest.raises(InputFileError, match="not found"):
        ChatterboxEngine(tmp_path / "nope.wav")


def test_unsupported_sample_extension_is_an_input_error(tmp_path):
    path = tmp_path / "voice.m4a"
    path.write_bytes(b"\0" * 100)

    with pytest.raises(InputFileError, match=r"\.wav"):
        ChatterboxEngine(path)


def test_unreadable_sample_is_an_input_error(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not audio at all")

    with pytest.raises(InputFileError, match="Could not read"):
        ChatterboxEngine(path)


def test_too_short_sample_is_rejected(tmp_path):
    path = write_sample(tmp_path / "short.wav", seconds=2.0)

    with pytest.raises(InvalidOptionError, match="at least 5 s"):
        ChatterboxEngine(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device": "tpu"},
        {"exaggeration": 0.1},
        {"exaggeration": 3.0},
        {"cfg_weight": -0.5},
        {"cfg_weight": 1.5},
        {"cfg_weight": True},
    ],
)
def test_invalid_options_are_rejected(sample, kwargs):
    with pytest.raises(InvalidOptionError):
        ChatterboxEngine(sample, **kwargs)


def test_other_voices_are_rejected_before_loading(sample):
    engine = ChatterboxEngine(sample)

    with pytest.raises(UnsupportedVoiceError, match="cloned voice"):
        engine.synthesize("Hello.", "af_heart")
    assert engine.device is None


def test_missing_dependencies_point_to_the_extra(sample, monkeypatch):
    monkeypatch.setitem(sys.modules, "chatterbox.tts", None)  # makes the import fail

    with pytest.raises(ModelLoadError, match="uv sync --extra clone"):
        ChatterboxEngine(sample, device="cpu").prepare(CLONE_VOICE_ID)


# --- synthesis with a stand-in model ---------------------------------------------------------


def test_synthesize_passes_the_style_options(sample):
    model = FakeModel()
    engine = loaded(ChatterboxEngine(sample, exaggeration=0.8, cfg_weight=0.3), model)

    audio = engine.synthesize("Hello there.", CLONE_VOICE_ID)

    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == 24_000
    assert model.calls == [("Hello there.", 0.8, 0.3)]


def test_speed_stretches_the_audio(sample):
    pytest.importorskip("librosa")
    engine = loaded(ChatterboxEngine(sample), FakeModel())

    faster = engine.synthesize("Hello there.", CLONE_VOICE_ID, speed=2.0)
    slower = engine.synthesize("Hello there.", CLONE_VOICE_ID, speed=0.5)

    assert len(faster) == pytest.approx(12_000, rel=0.05)
    assert len(slower) == pytest.approx(48_000, rel=0.05)


def test_model_failure_is_a_synthesis_error(sample):
    engine = loaded(ChatterboxEngine(sample), FakeModel(fail=True))

    with pytest.raises(SynthesisError, match="out of memory"):
        engine.synthesize("Hello.", CLONE_VOICE_ID)


def test_agent_generates_with_the_cloned_voice(sample, settings, tmp_path):
    model = FakeModel()
    engine = loaded(ChatterboxEngine(sample), model)
    text = "First sentence here. " * 30  # longer than one Chatterbox chunk

    result = TTSAgent(engine=engine, settings=settings).generate(
        text, output_path=tmp_path / "clone.wav"
    )

    assert result.voice == CLONE_VOICE_ID
    assert result.chunk_count == len(model.calls) > 1
    assert all(len(call[0]) <= MAX_CHUNK_CHARS for call in model.calls)
    assert result.subtitle_path is not None and result.subtitle_path.read_text(encoding="utf-8")
    assert sf.info(result.output_path).samplerate == 24_000


# --- agent integration -----------------------------------------------------------------------


def test_voice_sample_setting_selects_chatterbox(sample, settings):
    agent = TTSAgent(settings=replace(settings, voice_sample=sample, exaggeration=0.9))

    assert isinstance(agent.engine, ChatterboxEngine)
    assert agent.engine.exaggeration == 0.9
    assert agent.engine.device is None  # nothing loads until the first generate()


def test_default_voice_comes_from_the_engine_when_it_has_one(settings):
    class CloneLikeEngine(MockEngine):
        @property
        def default_voice(self):
            return "bf_emma"

    agent = TTSAgent(engine=CloneLikeEngine(), settings=settings)

    assert agent.resolve_voice("default") == "bf_emma"
    assert agent.resolve_voice(None) == "bf_emma"


def test_engine_chunk_limit_caps_the_setting(settings, tmp_path):
    class ShortChunkEngine(MockEngine):
        @property
        def max_chunk_chars(self):
            return 80

    engine = ShortChunkEngine()
    TTSAgent(engine=engine, settings=settings).generate(
        "A fairly ordinary sentence for testing. " * 10, output_path=tmp_path / "a.wav"
    )

    assert len(engine.calls) > 1
    assert all(len(text) <= 80 for text, _, _ in engine.calls)


def test_phoneme_pronunciations_are_skipped_for_plain_text_engines(settings, tmp_path):
    class PlainTextEngine(MockEngine):
        @property
        def supports_phonemes(self):
            return False

    dictionary = tmp_path / "pron.txt"
    dictionary.write_text("Kokoro = /kˈOkəɹO/\nSQL = sequel\n", encoding="utf-8")
    engine = PlainTextEngine()

    TTSAgent(engine=engine, settings=settings).generate(
        "Kokoro reads SQL.", output_path=tmp_path / "a.wav", pronunciations=dictionary
    )

    assert engine.calls[0][0] == "Kokoro reads sequel."


# --- real model ------------------------------------------------------------------------------


@pytest.mark.model
def test_real_chatterbox_clones_a_voice(tmp_path, settings):
    """Clone a Kokoro voice and check the result is speech at 24 kHz with subtitles."""
    pytest.importorskip("chatterbox")
    from tts_agent.engines.kokoro_engine import KokoroEngine

    reference = tmp_path / "reference.wav"
    kokoro = TTSAgent(engine=KokoroEngine(), settings=settings)
    kokoro.generate(
        "Alright, so here is the thing. I went down to the store this morning to grab some "
        "milk, and you will not believe what happened next. The man at the counter just laughed.",
        voice="am_fenrir",
        output_path=reference,
        subtitles=False,
    )

    result = TTSAgent(engine=ChatterboxEngine(reference), settings=settings).generate(
        "Hello there. This is a quick test of the cloned voice.",
        output_path=tmp_path / "clone.wav",
    )

    audio, rate = sf.read(result.output_path, dtype="float32")
    assert rate == 24_000
    assert 1.5 < result.duration_seconds < 15
    assert np.sqrt(np.mean(audio**2)) > 0.01
    assert result.subtitle_path is not None
