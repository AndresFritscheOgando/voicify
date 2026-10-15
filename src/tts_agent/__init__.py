"""Voicify: a local text-to-speech agent built on Kokoro."""

from tts_agent.agent import GenerationResult, TTSAgent
from tts_agent.config import Settings
from tts_agent.engines.base import TTSEngine, VoiceInfo
from tts_agent.errors import (
    EmptyTextError,
    InputFileError,
    InvalidOptionError,
    OutputError,
    SynthesisError,
    TTSError,
    UnsupportedVoiceError,
)
from tts_agent.logging_setup import setup_logging

__version__ = "0.1.0"

__all__ = [
    "EmptyTextError",
    "GenerationResult",
    "InputFileError",
    "InvalidOptionError",
    "OutputError",
    "Settings",
    "SynthesisError",
    "TTSAgent",
    "TTSEngine",
    "TTSError",
    "UnsupportedVoiceError",
    "VoiceInfo",
    "__version__",
    "setup_logging",
]
