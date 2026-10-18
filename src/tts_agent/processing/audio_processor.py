"""Audio output helpers."""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import numpy as np
import soundfile as sf

from tts_agent.errors import OutputError
from tts_agent.utils.file_utils import atomic_write

logger = logging.getLogger(__name__)

# Output formats the agent can write.
OUTPUT_FORMATS = ("wav", "mp3")

# MP3 bitrate. 192 kbps needs MPEG-1 sample rates (32 kHz and up); Kokoro's 24 kHz is MPEG-2,
# whose maximum is 160 kbps, so 24 kHz audio is encoded at 160 kbps instead of resampling it.
MP3_BITRATE_MPEG1 = 192
MP3_BITRATE_MPEG2 = 160
# LAME's encoder delay in samples (measured for lameenc / LAME 3.100: clicks line up exactly).
LAME_ENCODER_DELAY = 576


def duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    return len(audio) / sample_rate


def write_wav(audio: np.ndarray, sample_rate: int, path: Path) -> Path:
    """Write float mono audio as a 16-bit PCM WAV. Samples outside [-1, 1] are clipped."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1:
        raise OutputError(f"Expected mono audio (1-D array), got shape {samples.shape}")
    if samples.size == 0:
        raise OutputError("Refusing to write an empty WAV file")
    path = Path(path)
    try:
        clipped = np.clip(samples, -1.0, 1.0)
        atomic_write(
            path, lambda tmp: sf.write(tmp, clipped, sample_rate, subtype="PCM_16", format="WAV")
        )
    except (OSError, RuntimeError) as exc:
        raise OutputError(f"Could not write WAV file {path}: {exc}") from exc
    return path


def write_audio(audio: np.ndarray, sample_rate: int, path: Path, format: str) -> Path:
    """Write `audio` as `format` ("wav" or "mp3")."""
    if format == "wav":
        return write_wav(audio, sample_rate, path)
    if format == "mp3":
        return write_mp3(audio, sample_rate, path)
    raise OutputError(f"Unsupported output format {format!r}")


def write_mp3(audio: np.ndarray, sample_rate: int, path: Path) -> Path:
    """Encode float mono audio as a CBR MP3 with a gapless (LAME "Info") header."""
    path = Path(path)
    data = encode_mp3(audio, sample_rate)
    try:
        atomic_write(path, lambda tmp: tmp.write_bytes(data))
    except OSError as exc:
        raise OutputError(f"Could not write MP3 file {path}: {exc}") from exc
    return path


def encode_mp3(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float mono audio to MP3 bytes with lameenc (no ffmpeg needed).

    lameenc writes bare frames, so decoders would keep LAME's start delay and end padding
    (about 50-70 ms at 24 kHz). A leading "Info" frame with the LAME gapless fields tells
    decoders such as mpg123 and ffmpeg to trim them, so the MP3 decodes to the exact WAV length.
    """
    try:
        import lameenc
    except ImportError as exc:
        raise OutputError(
            "MP3 output needs the 'lameenc' package, which is not installed. "
            "Install it with `uv add lameenc` (or `pip install lameenc`), or use --format wav."
        ) from exc

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1:
        raise OutputError(f"Expected mono audio (1-D array), got shape {samples.shape}")
    if samples.size == 0:
        raise OutputError("Refusing to write an empty MP3 file")

    bitrate = MP3_BITRATE_MPEG1 if sample_rate >= 32_000 else MP3_BITRATE_MPEG2
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).round().astype("<i2").tobytes()
    try:
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(bitrate)
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(1)
        encoder.set_quality(2)  # 2 = high quality, the usual choice for final output
        frames = bytes(encoder.encode(pcm)) + bytes(encoder.flush())
    except Exception as exc:
        raise OutputError(f"MP3 encoding failed: {exc}") from exc
    return _with_gapless_header(frames, len(samples))


# MPEG audio frame header tables (Layer III only).
_MP3_BITRATES = {
    "1": (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    "2": (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}
_MP3_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}


def _mp3_frames(data: bytes) -> list[tuple[int, int]]:
    """Return (offset, size) of every Layer III frame in a bare MP3 stream."""
    frames = []
    i = 0
    while i + 4 <= len(data):
        header = int.from_bytes(data[i : i + 4], "big")
        version = (header >> 19) & 3
        if (header >> 21) & 0x7FF != 0x7FF or version == 1 or (header >> 17) & 3 != 1:
            raise OutputError(f"Unexpected MP3 frame header at byte {i}")
        mpeg1 = version == 3
        bitrate = _MP3_BITRATES["1" if mpeg1 else "2"][(header >> 12) & 0xF] * 1000
        rate = _MP3_RATES[version][(header >> 10) & 3]
        size = (144 if mpeg1 else 72) * bitrate // rate + ((header >> 9) & 1)
        if size <= 4:
            raise OutputError(f"Invalid MP3 frame size at byte {i}")
        frames.append((i, size))
        i += size
    return frames


def _with_gapless_header(frames: bytes, sample_count: int) -> bytes:
    """Prepend a Xing "Info" frame carrying LAME's encoder delay and end padding."""
    layout = _mp3_frames(frames)
    if not layout:
        return frames
    first_offset, first_size = layout[0]
    header = frames[first_offset : first_offset + 4]
    version = (header[1] >> 3) & 3
    mono = (header[3] >> 6) & 3 == 3
    bitrate_kbps = _MP3_BITRATES["1" if version == 3 else "2"][header[2] >> 4]
    samples_per_frame = 1152 if version == 3 else 576
    side_info = (17 if mono else 32) if version == 3 else (9 if mono else 17)
    padding = len(layout) * samples_per_frame - LAME_ENCODER_DELAY - sample_count
    tag_start = 4 + side_info
    if not 0 <= padding < 4096 or first_size < tag_start + 40:
        logger.warning("mp3_gapless_header_skipped padding=%d", padding)
        return frames

    info = bytearray(first_size)
    info[:4] = header
    info[tag_start : tag_start + 4] = b"Info"  # "Info" marks a CBR stream ("Xing" is for VBR)
    struct.pack_into(">III", info, tag_start + 4, 0x03, len(layout), len(frames) + first_size)
    lame = tag_start + 16  # LAME extension follows the frame and byte counts
    info[lame : lame + 9] = b"LAME3.100"
    info[lame + 9] = 0x01  # tag revision 0, CBR
    info[lame + 20] = min(255, bitrate_kbps)
    info[lame + 21 : lame + 24] = ((LAME_ENCODER_DELAY << 12) | padding).to_bytes(3, "big")
    return bytes(info) + frames


SILENCE_THRESHOLD = 0.01  # RMS over 5 ms windows, about -40 dBFS
_WINDOW_MS = 5
_FADE_MS = 5


def leading_silence(audio: np.ndarray, sample_rate: int) -> int:
    """Samples of near-silence at the start of `audio`, in whole 5 ms windows."""
    return _silent_windows(audio, sample_rate) * _window(sample_rate)


def trailing_silence(audio: np.ndarray, sample_rate: int) -> int:
    """Samples of near-silence at the end of `audio`, in whole 5 ms windows."""
    window = _window(sample_rate)
    remainder = len(audio) % window
    windows = _silent_windows(audio[remainder:][::-1], sample_rate)
    silent = windows * window
    return silent + remainder if silent == len(audio) - remainder else silent


def fade(audio: np.ndarray, sample_rate: int, fade_in: bool) -> np.ndarray:
    """A 5 ms linear fade at a cut edge, so trimming never clicks."""
    n = min(len(audio), round(sample_rate * _FADE_MS / 1000))
    if n == 0:
        return audio
    out = np.array(audio, dtype=np.float32, copy=True)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if fade_in:
        out[:n] *= ramp
    else:
        out[-n:] *= ramp[::-1]
    return out


def _window(sample_rate: int) -> int:
    return max(1, round(sample_rate * _WINDOW_MS / 1000))


def _silent_windows(audio: np.ndarray, sample_rate: int) -> int:
    window = _window(sample_rate)
    count = len(audio) // window
    if count == 0:
        return 0
    frames = np.asarray(audio[: count * window], dtype=np.float32).reshape(count, window)
    loud = np.sqrt(np.mean(frames**2, axis=1)) > SILENCE_THRESHOLD
    first_loud = int(np.argmax(loud)) if loud.any() else count
    return first_loud


def silence(milliseconds: int, sample_rate: int) -> np.ndarray:
    """A block of float32 silence."""
    return np.zeros(round(sample_rate * milliseconds / 1000), dtype=np.float32)


def concatenate(segments: list[np.ndarray]) -> np.ndarray:
    """Join audio segments into one float32 array in a single copy."""
    if not segments:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([np.asarray(s, dtype=np.float32).reshape(-1) for s in segments])
