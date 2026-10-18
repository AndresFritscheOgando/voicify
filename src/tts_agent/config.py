"""Runtime settings, read from environment variables and an optional `.env` file."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from tts_agent.errors import InvalidOptionError

DEVICES = ("auto", "cuda", "cpu")
CHUNK_ERROR_POLICIES = ("skip", "abort")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
MIN_CHUNK_CHARS = 50
MIN_SPEED = 0.5
MAX_SPEED = 2.0
DEFAULT_VOICE_ALIAS = "default"
MIN_EXAGGERATION = 0.25
MAX_EXAGGERATION = 2.0
DEFAULT_EXAGGERATION = 0.5
DEFAULT_CFG_WEIGHT = 0.5
MAX_SEED = 2**32 - 1
MAX_PAUSE_MS = 60_000  # longest [pause] or configured pause; a typo like "1500s" stays sane

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Settings:
    """Agent configuration. See `.env.example` for what each value means."""

    device: str = "auto"
    default_voice: str = "af_heart"
    default_speed: float = 1.0
    output_dir: Path = Path("outputs")
    max_chunk_chars: int = 400
    sentence_pause_ms: int = 250
    paragraph_pause_ms: int = 700
    chunk_retries: int = 1
    on_chunk_error: str = "skip"
    pronunciations: Path | None = None
    subtitles: bool = True
    normalize: bool = True
    log_level: str = "INFO"
    voice_sample: Path | None = None
    exaggeration: float = DEFAULT_EXAGGERATION
    cfg_weight: float = DEFAULT_CFG_WEIGHT
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.device not in DEVICES:
            raise InvalidOptionError(
                f"TTS_DEVICE must be one of {', '.join(DEVICES)}, got {self.device!r}"
            )
        if self.on_chunk_error not in CHUNK_ERROR_POLICIES:
            raise InvalidOptionError(
                f"TTS_ON_CHUNK_ERROR must be one of {', '.join(CHUNK_ERROR_POLICIES)}, "
                f"got {self.on_chunk_error!r}"
            )
        if self.log_level not in LOG_LEVELS:
            raise InvalidOptionError(
                f"TTS_LOG_LEVEL must be one of {', '.join(LOG_LEVELS)}, got {self.log_level!r}"
            )
        validate_speed(self.default_speed, name="TTS_DEFAULT_SPEED")
        validate_exaggeration(self.exaggeration, name="TTS_EXAGGERATION")
        validate_cfg_weight(self.cfg_weight, name="TTS_CFG_WEIGHT")
        if self.seed is not None:
            validate_seed(self.seed, name="TTS_SEED")
        for name, value in (
            ("TTS_SENTENCE_PAUSE_MS", self.sentence_pause_ms),
            ("TTS_PARAGRAPH_PAUSE_MS", self.paragraph_pause_ms),
        ):
            if not 0 <= value <= MAX_PAUSE_MS:
                raise InvalidOptionError(
                    f"{name} must be between 0 and {MAX_PAUSE_MS} ms, got {value}"
                )
        if self.max_chunk_chars < MIN_CHUNK_CHARS:
            raise InvalidOptionError(
                f"TTS_MAX_CHUNK_CHARS must be at least {MIN_CHUNK_CHARS}, "
                f"got {self.max_chunk_chars}"
            )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from `environ`, or from `os.environ` plus `.env` when it is None.

        `.env` never overrides variables that are already set in the environment. It is loaded
        into `os.environ` so that Hugging Face variables such as `HF_HUB_OFFLINE` take effect too.
        """
        if environ is None:
            # Search from the working directory: the default searches from this file's folder,
            # which would miss the user's .env once the package is installed.
            load_dotenv(find_dotenv(usecwd=True), override=False)
            environ = os.environ

        def get(name: str) -> str | None:
            value = environ.get(name)
            return value.strip() if value is not None and value.strip() else None

        kwargs: dict[str, object] = {}
        if (v := get("TTS_DEVICE")) is not None:
            kwargs["device"] = v.lower()
        if (v := get("TTS_DEFAULT_VOICE")) is not None:
            kwargs["default_voice"] = v
        if (v := get("TTS_DEFAULT_SPEED")) is not None:
            kwargs["default_speed"] = _parse_float("TTS_DEFAULT_SPEED", v)
        if (v := get("TTS_OUTPUT_DIR")) is not None:
            kwargs["output_dir"] = Path(v)
        if (v := get("TTS_MAX_CHUNK_CHARS")) is not None:
            kwargs["max_chunk_chars"] = _parse_int("TTS_MAX_CHUNK_CHARS", v)
        if (v := get("TTS_SENTENCE_PAUSE_MS")) is not None:
            kwargs["sentence_pause_ms"] = _parse_int("TTS_SENTENCE_PAUSE_MS", v)
        if (v := get("TTS_PARAGRAPH_PAUSE_MS")) is not None:
            kwargs["paragraph_pause_ms"] = _parse_int("TTS_PARAGRAPH_PAUSE_MS", v)
        if (v := get("TTS_CHUNK_RETRIES")) is not None:
            kwargs["chunk_retries"] = _parse_int("TTS_CHUNK_RETRIES", v)
        if (v := get("TTS_ON_CHUNK_ERROR")) is not None:
            kwargs["on_chunk_error"] = v.lower()
        if (v := get("TTS_PRONUNCIATIONS")) is not None:
            kwargs["pronunciations"] = Path(v)
        if (v := get("TTS_SUBTITLES")) is not None:
            kwargs["subtitles"] = _parse_bool("TTS_SUBTITLES", v)
        if (v := get("TTS_NORMALIZE")) is not None:
            kwargs["normalize"] = _parse_bool("TTS_NORMALIZE", v)
        if (v := get("TTS_LOG_LEVEL")) is not None:
            kwargs["log_level"] = v.upper()
        if (v := get("TTS_VOICE_SAMPLE")) is not None:
            kwargs["voice_sample"] = Path(v)
        if (v := get("TTS_EXAGGERATION")) is not None:
            kwargs["exaggeration"] = _parse_float("TTS_EXAGGERATION", v)
        if (v := get("TTS_CFG_WEIGHT")) is not None:
            kwargs["cfg_weight"] = _parse_float("TTS_CFG_WEIGHT", v)
        if (v := get("TTS_SEED")) is not None:
            kwargs["seed"] = _parse_int("TTS_SEED", v)
        return cls(**kwargs)  # type: ignore[arg-type]


def validate_speed(speed: object, name: str = "speed") -> float:
    """Return `speed` as a float, or raise `InvalidOptionError` if it is outside 0.5-2.0."""
    if isinstance(speed, bool) or not isinstance(speed, int | float):
        raise InvalidOptionError(f"{name} must be a number, got {speed!r}")
    if not MIN_SPEED <= speed <= MAX_SPEED:  # also rejects NaN
        raise InvalidOptionError(f"{name} must be between {MIN_SPEED} and {MAX_SPEED}, got {speed}")
    return float(speed)


def validate_exaggeration(value: object, name: str = "exaggeration") -> float:
    """Return `value` as a float, or raise `InvalidOptionError` if it is outside 0.25-2.0."""
    return _validate_range(value, name, MIN_EXAGGERATION, MAX_EXAGGERATION)


def validate_cfg_weight(value: object, name: str = "cfg_weight") -> float:
    """Return `value` as a float, or raise `InvalidOptionError` if it is outside 0-1."""
    return _validate_range(value, name, 0.0, 1.0)


def validate_seed(value: object, name: str = "seed") -> int:
    """Return `value` as an int, or raise `InvalidOptionError` unless it is 0 to 2**32 - 1."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidOptionError(f"{name} must be a whole number, got {value!r}")
    if not 0 <= value <= MAX_SEED:
        raise InvalidOptionError(f"{name} must be between 0 and {MAX_SEED}, got {value}")
    return value


def _validate_range(value: object, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidOptionError(f"{name} must be a number, got {value!r}")
    if not low <= value <= high:  # also rejects NaN
        raise InvalidOptionError(f"{name} must be between {low} and {high}, got {value}")
    return float(value)


def _parse_int(name: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise InvalidOptionError(f"{name} must be an integer, got {value!r}") from None
    if parsed < 0:
        raise InvalidOptionError(f"{name} must not be negative, got {parsed}")
    return parsed


def _parse_float(name: str, value: str) -> float:
    try:
        return float(value)
    except ValueError:
        raise InvalidOptionError(f"{name} must be a number, got {value!r}") from None


def _parse_bool(name: str, value: str) -> bool:
    lowered = value.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise InvalidOptionError(f"{name} must be true or false, got {value!r}")
