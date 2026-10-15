from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from tts_agent.errors import OutputError
from tts_agent.processing.audio_processor import duration_seconds, write_wav


def test_write_wav_is_16bit_pcm_mono_24k(tmp_path):
    audio = np.linspace(-0.5, 0.5, 24_000, dtype=np.float32)

    path = write_wav(audio, 24_000, tmp_path / "a.wav")

    info = sf.info(path)
    assert info.samplerate == 24_000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert info.frames == 24_000
    read, _ = sf.read(path, dtype="float32")
    np.testing.assert_allclose(read, audio, atol=1 / 32_768 * 2)


def test_write_wav_clips_out_of_range_samples(tmp_path):
    path = write_wav(np.array([-2.0, 0.0, 2.0], dtype=np.float32), 24_000, tmp_path / "a.wav")

    read, _ = sf.read(path, dtype="float32")
    assert read.min() >= -1.0
    assert read.max() <= 1.0


def test_write_wav_rejects_empty_audio(tmp_path):
    with pytest.raises(OutputError):
        write_wav(np.zeros(0, dtype=np.float32), 24_000, tmp_path / "a.wav")


def test_write_wav_rejects_multichannel_audio(tmp_path):
    with pytest.raises(OutputError):
        write_wav(np.zeros((10, 2), dtype=np.float32), 24_000, tmp_path / "a.wav")


def test_write_wav_reports_unwritable_path(tmp_path):
    with pytest.raises(OutputError):
        write_wav(np.zeros(10, dtype=np.float32), 24_000, tmp_path / "missing" / "a.wav")


def test_duration_seconds():
    assert duration_seconds(np.zeros(36_000), 24_000) == pytest.approx(1.5)
