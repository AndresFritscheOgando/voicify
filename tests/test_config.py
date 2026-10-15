from __future__ import annotations

from pathlib import Path

import pytest

from tts_agent.config import Settings
from tts_agent.errors import InvalidOptionError


def test_defaults_with_empty_environment():
    settings = Settings.from_env({})

    assert settings == Settings()
    assert settings.device == "auto"
    assert settings.default_voice == "af_heart"
    assert settings.output_dir == Path("outputs")
    assert settings.pronunciations is None
    assert settings.subtitles is True


def test_values_are_read_and_parsed():
    settings = Settings.from_env(
        {
            "TTS_DEVICE": "CPU",
            "TTS_DEFAULT_VOICE": "am_michael",
            "TTS_DEFAULT_SPEED": "1.25",
            "TTS_OUTPUT_DIR": "renders",
            "TTS_MAX_CHUNK_CHARS": "300",
            "TTS_SENTENCE_PAUSE_MS": "100",
            "TTS_PARAGRAPH_PAUSE_MS": "900",
            "TTS_CHUNK_RETRIES": "2",
            "TTS_ON_CHUNK_ERROR": "abort",
            "TTS_PRONUNCIATIONS": "pron.txt",
            "TTS_SUBTITLES": "false",
            "TTS_LOG_LEVEL": "debug",
        }
    )

    assert settings == Settings(
        device="cpu",
        default_voice="am_michael",
        default_speed=1.25,
        output_dir=Path("renders"),
        max_chunk_chars=300,
        sentence_pause_ms=100,
        paragraph_pause_ms=900,
        chunk_retries=2,
        on_chunk_error="abort",
        pronunciations=Path("pron.txt"),
        subtitles=False,
        log_level="DEBUG",
    )


def test_blank_values_fall_back_to_defaults():
    assert Settings.from_env({"TTS_PRONUNCIATIONS": "", "TTS_DEVICE": "  "}) == Settings()


@pytest.mark.parametrize(
    "env",
    [
        {"TTS_DEVICE": "tpu"},
        {"TTS_ON_CHUNK_ERROR": "ignore"},
        {"TTS_LOG_LEVEL": "loud"},
        {"TTS_MAX_CHUNK_CHARS": "lots"},
        {"TTS_MAX_CHUNK_CHARS": "49"},
        {"TTS_CHUNK_RETRIES": "-1"},
        {"TTS_DEFAULT_SPEED": "fast"},
        {"TTS_DEFAULT_SPEED": "0.4"},
        {"TTS_DEFAULT_SPEED": "2.5"},
        {"TTS_SUBTITLES": "maybe"},
    ],
)
def test_invalid_values_raise(env):
    with pytest.raises(InvalidOptionError):
        Settings.from_env(env)
