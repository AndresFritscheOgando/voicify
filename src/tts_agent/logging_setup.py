"""Logging configuration. Messages use a `key=value` style so they are easy to grep."""

from __future__ import annotations

import logging
import warnings

LOGGER_NAME = "tts_agent"
_FORMAT = "%(asctime)s level=%(levelname)s logger=%(name)s %(message)s"
_HANDLER_MARKER = "_tts_agent_handler"

# Harmless warnings that Kokoro and torch print while the model loads.
_NOISY_WARNINGS = (
    r".*torch\.nn\.utils\.weight_norm.*is deprecated.*",
    r".*torch\.jit\.script.*",
    r".*dropout option adds dropout after all but last recurrent layer.*",
)


def setup_logging(level: str | int = "INFO") -> logging.Logger:
    """Configure the `tts_agent` logger. Safe to call more than once."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not any(getattr(h, _HANDLER_MARKER, False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)
        logger.propagate = False
    silence_model_warnings()
    return logger


def silence_model_warnings() -> None:
    """Hide the known-harmless deprecation warnings emitted while Kokoro loads."""
    for pattern in _NOISY_WARNINGS:
        warnings.filterwarnings("ignore", message=pattern)
