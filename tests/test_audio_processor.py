from __future__ import annotations

import sys

import numpy as np
import pytest
import soundfile as sf

from tts_agent.errors import OutputError
from tts_agent.processing.audio_processor import (
    _MP3_BITRATES,
    concatenate,
    duration_seconds,
    encode_mp3,
    silence,
    write_audio,
    write_mp3,
    write_wav,
)


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


# --- MP3 -------------------------------------------------------------------------------------


def clicks(n: int, positions: list[int]) -> np.ndarray:
    audio = np.zeros(n, dtype=np.float32)
    for p in positions:
        audio[p : p + 24] = np.hanning(24) * 0.8
    return audio


def click_positions(audio: np.ndarray, count: int) -> list[int]:
    envelope = np.abs(audio).copy()
    found = []
    for _ in range(count):
        i = int(np.argmax(envelope))
        found.append(i - 12)  # the peak of a 24-sample bump is its middle
        envelope[max(0, i - 200) : i + 200] = 0
    return sorted(found)


@pytest.mark.parametrize("rate", [24_000, 44_100])
@pytest.mark.parametrize("seconds", [0.05, 3.0003, 30.0])
def test_mp3_decodes_to_the_exact_length_and_alignment(tmp_path, rate, seconds):
    n = round(rate * seconds)
    positions = [100, n // 2, n - 200]
    path = write_mp3(clicks(n, positions), rate, tmp_path / "a.mp3")

    decoded, decoded_rate = sf.read(path, dtype="float32")

    assert decoded_rate == rate
    assert len(decoded) == n  # the gapless header trims LAME's delay and padding
    for found, expected in zip(click_positions(decoded, 3), positions, strict=True):
        assert abs(found - expected) <= 2


def test_mp3_is_cbr_mono_at_the_expected_bitrate():
    for rate, kbps in ((24_000, 160), (44_100, 192)):
        data = encode_mp3(np.zeros(rate, dtype=np.float32), rate)
        header = data[:4]
        assert header[0] == 0xFF and header[1] & 0xE0 == 0xE0
        assert (header[3] >> 6) & 3 == 3  # mono
        version = (header[1] >> 3) & 3
        table = _MP3_BITRATES["1" if version == 3 else "2"]
        assert table[header[2] >> 4] == kbps
        assert b"Info" in data[:64] and b"LAME" in data[:128]


def test_mp3_waveform_survives_encoding(tmp_path):
    t = np.arange(24_000) / 24_000
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    decoded, _ = sf.read(write_mp3(audio, 24_000, tmp_path / "a.mp3"), dtype="float32")

    correlation = np.corrcoef(audio, decoded)[0, 1]
    assert correlation > 0.99


def test_mp3_rejects_empty_or_multichannel_audio(tmp_path):
    with pytest.raises(OutputError):
        write_mp3(np.zeros(0, dtype=np.float32), 24_000, tmp_path / "a.mp3")
    with pytest.raises(OutputError):
        write_mp3(np.zeros((10, 2), dtype=np.float32), 24_000, tmp_path / "a.mp3")


def test_missing_lameenc_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "lameenc", None)  # makes `import lameenc` fail

    with pytest.raises(OutputError, match="lameenc"):
        write_mp3(np.zeros(100, dtype=np.float32), 24_000, tmp_path / "a.mp3")
    assert not (tmp_path / "a.mp3").exists()


def test_write_audio_dispatches_by_format(tmp_path):
    audio = np.zeros(2400, dtype=np.float32)

    assert sf.info(write_audio(audio, 24_000, tmp_path / "a.wav", "wav")).format == "WAV"
    assert sf.info(write_audio(audio, 24_000, tmp_path / "a.mp3", "mp3")).format == "MP3"
    with pytest.raises(OutputError):
        write_audio(audio, 24_000, tmp_path / "a.ogg", "ogg")


# --- concatenation and silence ---------------------------------------------------------------


def test_concatenate_keeps_order_and_length():
    parts = [np.full(10, 0.1), np.full(5, 0.2), np.full(7, 0.3)]

    joined = concatenate(parts)

    assert joined.dtype == np.float32
    assert len(joined) == 22
    np.testing.assert_allclose(joined[:10], 0.1)
    np.testing.assert_allclose(joined[10:15], 0.2)
    np.testing.assert_allclose(joined[15:], 0.3)


def test_concatenate_flattens_and_handles_empty_input():
    assert len(concatenate([])) == 0
    assert concatenate([np.zeros((2, 3)), np.zeros(0)]).shape == (6,)


@pytest.mark.parametrize(("ms", "samples"), [(0, 0), (250, 6000), (700, 16800), (1, 24)])
def test_silence_length(ms, samples):
    block = silence(ms, 24_000)

    assert len(block) == samples
    assert not block.any()


def test_mp3_reports_unwritable_path(tmp_path):
    with pytest.raises(OutputError, match="Could not write MP3"):
        write_mp3(np.zeros(2400, dtype=np.float32), 24_000, tmp_path)  # a directory
