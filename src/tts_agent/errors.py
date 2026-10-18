"""Exception hierarchy. Every error the agent raises on purpose is a `TTSError`."""

from __future__ import annotations


class TTSError(Exception):
    """Base class for all errors raised by the TTS agent."""


class EmptyTextError(TTSError):
    """The input text is empty or contains only whitespace."""


class InputFileError(TTSError):
    """An input file is missing, unreadable, not UTF-8, or has the wrong extension."""


class UnsupportedVoiceError(TTSError):
    """The requested voice is not available in the engine's catalog."""


class InvalidOptionError(TTSError):
    """An option has an invalid value (device, speed, output format, ...)."""


class SynthesisError(TTSError):
    """The engine failed to load or to produce audio."""


class ModelLoadError(SynthesisError):
    """The engine's model could not be loaded. Never retried per chunk."""


class ChunkingError(TTSError):
    """Splitting the text into chunks lost, duplicated or reordered text. Always a bug."""


class OutputError(TTSError):
    """The output file could not be created or written."""
