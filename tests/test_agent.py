from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from tts_agent import (
    EmptyTextError,
    InvalidOptionError,
    Settings,
    SynthesisError,
    TTSAgent,
    UnsupportedVoiceError,
)
from tts_agent.engines.mock_engine import MockEngine

EXAMPLE = "Welcome to our channel. Today we are going to explore local text-to-speech."


def test_generate_writes_wav(agent, tmp_path):
    result = agent.generate(EXAMPLE, output_path=tmp_path / "hello.wav")

    assert result.output_path == tmp_path / "hello.wav"
    audio, rate = sf.read(result.output_path, dtype="float32")
    assert rate == 24_000
    assert audio.ndim == 1
    assert np.abs(audio).max() > 0.1
    assert result.duration_seconds == pytest.approx(len(audio) / rate)
    assert result.sample_rate == 24_000
    assert result.chunk_count == 1
    assert result.failed_chunks == []
    assert result.format == "wav"


def test_generate_uses_defaults(agent, mock_engine, settings):
    result = agent.generate(EXAMPLE)

    assert result.output_path == settings.output_dir / "narration.wav"
    assert result.output_path.exists()
    assert result.voice == "af_heart"
    assert mock_engine.calls == [(EXAMPLE, "af_heart", 1.0)]


def test_generate_passes_voice_and_speed(agent, mock_engine, tmp_path):
    result = agent.generate(EXAMPLE, voice="am_michael", speed=1.5, output_path=tmp_path / "a.wav")

    assert result.voice == "am_michael"
    assert mock_engine.calls == [(EXAMPLE, "am_michael", 1.5)]


@pytest.mark.parametrize("text", ["", "   ", "\n\t \n"])
def test_empty_text_raises(agent, mock_engine, settings, text):
    with pytest.raises(EmptyTextError):
        agent.generate(text)
    assert mock_engine.calls == []
    assert not settings.output_dir.exists()


def test_missing_output_directories_are_created(agent, tmp_path):
    target = tmp_path / "deep" / "nested" / "dir" / "out.wav"

    result = agent.generate(EXAMPLE, output_path=target)

    assert result.output_path == target
    assert target.exists()


def test_existing_file_is_not_overwritten(agent, tmp_path):
    target = tmp_path / "narration.wav"
    target.write_bytes(b"keep me")

    first = agent.generate(EXAMPLE, output_path=target)
    second = agent.generate(EXAMPLE, output_path=target)

    assert target.read_bytes() == b"keep me"
    assert first.output_path == tmp_path / "narration_1.wav"
    assert second.output_path == tmp_path / "narration_2.wav"


def test_overwrite_replaces_existing_file(agent, tmp_path):
    target = tmp_path / "narration.wav"
    target.write_bytes(b"old")

    result = agent.generate(EXAMPLE, output_path=target, overwrite=True)

    assert result.output_path == target
    assert target.read_bytes() != b"old"


def test_output_path_without_extension_gets_wav(agent, tmp_path):
    result = agent.generate(EXAMPLE, output_path=tmp_path / "clip")

    assert result.output_path == tmp_path / "clip.wav"


def test_unsupported_output_extension(agent, tmp_path):
    with pytest.raises(InvalidOptionError):
        agent.generate(EXAMPLE, output_path=tmp_path / "clip.mp3")


def test_engine_returning_no_audio_raises(settings, tmp_path):
    class EmptyEngine(MockEngine):
        def synthesize(self, text, voice, speed=1.0):
            return np.zeros(0, dtype=np.float32)

    agent = TTSAgent(engine=EmptyEngine(), settings=settings)
    with pytest.raises(SynthesisError):
        agent.generate(EXAMPLE, output_path=tmp_path / "x.wav")
    assert not (tmp_path / "x.wav").exists()


def test_list_voices_comes_from_engine(agent, mock_engine):
    assert agent.list_voices() == mock_engine.list_voices()


@pytest.mark.parametrize("voice", ["default", None, "DEFAULT", " default "])
def test_default_alias_uses_configured_voice(mock_engine, tmp_path, voice):
    agent = TTSAgent(engine=mock_engine, settings=Settings(default_voice="am_michael"))

    result = agent.generate(EXAMPLE, voice=voice, output_path=tmp_path / "a.wav")

    assert result.voice == "am_michael"
    assert mock_engine.calls[-1][1] == "am_michael"


def test_voice_ids_are_case_insensitive(agent, tmp_path):
    assert agent.generate(EXAMPLE, voice="BF_Emma", output_path=tmp_path / "a.wav").voice == (
        "bf_emma"
    )


def test_unsupported_voice_suggests_close_matches(agent, mock_engine, settings):
    with pytest.raises(UnsupportedVoiceError) as excinfo:
        agent.generate(EXAMPLE, voice="af_hart")

    message = str(excinfo.value)
    assert "'af_hart'" in message
    assert "Did you mean: af_heart" in message
    assert "list_voices()" in message
    assert mock_engine.calls == []
    assert not settings.output_dir.exists()


def test_unsupported_voice_without_close_match(agent):
    with pytest.raises(UnsupportedVoiceError) as excinfo:
        agent.generate(EXAMPLE, voice="zz_nobody")

    assert "Did you mean" not in str(excinfo.value)


def test_bad_default_voice_names_the_setting(mock_engine):
    agent = TTSAgent(engine=mock_engine, settings=Settings(default_voice="af_nope"))

    with pytest.raises(UnsupportedVoiceError, match="TTS_DEFAULT_VOICE"):
        agent.generate(EXAMPLE)


@pytest.mark.parametrize("speed", [0.5, 1, 1.25, 2.0])
def test_valid_speeds_are_passed_to_engine(agent, mock_engine, tmp_path, speed):
    agent.generate(EXAMPLE, speed=speed, output_path=tmp_path / "a.wav")

    assert mock_engine.calls[-1][2] == speed


def test_default_speed_comes_from_settings(mock_engine, tmp_path):
    agent = TTSAgent(engine=mock_engine, settings=Settings(default_speed=1.5))

    agent.generate(EXAMPLE, output_path=tmp_path / "a.wav")

    assert mock_engine.calls[-1][2] == 1.5


@pytest.mark.parametrize("speed", [0.49, 2.01, 0, -1, float("nan"), float("inf"), "fast", True])
def test_invalid_speed_is_rejected(agent, mock_engine, speed):
    with pytest.raises(InvalidOptionError, match="speed"):
        agent.generate(EXAMPLE, speed=speed)
    assert mock_engine.calls == []


def test_faster_speed_gives_shorter_audio(agent, tmp_path):
    slow = agent.generate(EXAMPLE, speed=0.5, output_path=tmp_path / "slow.wav")
    fast = agent.generate(EXAMPLE, speed=2.0, output_path=tmp_path / "fast.wav")

    assert fast.duration_seconds < slow.duration_seconds
