"""Audio output helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from tts_agent.errors import OutputError


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
        sf.write(path, np.clip(samples, -1.0, 1.0), sample_rate, subtype="PCM_16", format="WAV")
    except (OSError, RuntimeError) as exc:
        raise OutputError(f"Could not write WAV file {path}: {exc}") from exc
    return path
