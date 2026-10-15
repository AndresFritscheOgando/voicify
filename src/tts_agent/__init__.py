"""Voicify: a local text-to-speech agent built on Kokoro."""

from tts_agent.agent import ChunkTiming, GenerationResult, TTSAgent
from tts_agent.config import Settings
from tts_agent.engines.base import TTSEngine, VoiceInfo
from tts_agent.errors import (
    EmptyTextError,
    InputFileError,
    InvalidOptionError,
    ModelLoadError,
    OutputError,
    SynthesisError,
    TTSError,
    UnsupportedVoiceError,
)
from tts_agent.logging_setup import setup_logging

__version__ = "0.1.0"

__all__ = [
    "ChunkTiming",
    "EmptyTextError",
    "GenerationResult",
    "InputFileError",
    "InvalidOptionError",
    "ModelLoadError",
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
