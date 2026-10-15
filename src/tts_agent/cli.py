"""Command-line interface: `voicify ...` or `python main.py ...`."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import logging
import sys
import traceback
from pathlib import Path
from typing import TextIO

from tts_agent import __version__
from tts_agent.agent import TTSAgent
from tts_agent.config import DEVICES, Settings
from tts_agent.engines.base import TTSEngine, VoiceInfo
from tts_agent.errors import (
    EmptyTextError,
    InputFileError,
    InvalidOptionError,
    TTSError,
    UnsupportedVoiceError,
)
from tts_agent.logging_setup import setup_logging
from tts_agent.processing.audio_processor import OUTPUT_FORMATS

EXIT_OK = 0
EXIT_GENERATION_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_PARTIAL = 3
EXIT_INTERRUPTED = 130

# Errors caused by the user's input or options rather than by generation itself.
_USAGE_ERRORS = (EmptyTextError, InputFileError, InvalidOptionError, UnsupportedVoiceError)

EXAMPLES = """\
examples:
  voicify --text "Hello, welcome to my channel." --output outputs/hello.wav
  voicify --file scripts/example.txt --voice am_michael --speed 0.9
  voicify --file scripts/example.txt --format mp3 --keep-wav
  voicify --file script.txt --pronunciations pronunciations.txt
  voicify --file script.txt --voice-sample samples/narrator.wav --exaggeration 0.7
  voicify --list-voices

voice cloning (needs `uv sync --extra clone`):
  --voice-sample speaks in the voice of a 5+ s clip of clean speech, using Chatterbox instead
  of Kokoro. --exaggeration (0.25-2.0) makes it more expressive; --cfg-weight (0-1) lower
  makes it slower and more deliberate. Subtitles are timed by character count.

script controls:
  [pause] (1 s), [pause 1.5s] or [pause 800ms] anywhere in the text inserts an exact silence.
  Blank lines between paragraphs get a longer pause (TTS_PARAGRAPH_PAUSE_MS).

exit codes:
  0 success, 1 generation error, 2 usage or validation error,
  3 partial success (some chunks failed and were skipped)
"""

logger = logging.getLogger(__name__)


class _ArgumentParser(argparse.ArgumentParser):
    """argparse that reports usage errors with our exit code and one-line style."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE_ERROR, f"error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="voicify",
        description="Local text-to-speech for long-form narration, powered by Kokoro-82M, "
        "with optional voice cloning through Chatterbox.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="text to speak")
    source.add_argument("--file", metavar="PATH", help="UTF-8 .txt script to speak")
    source.add_argument(
        "--list-voices", action="store_true", help="print the available voices and exit"
    )
    parser.add_argument(
        "--voice", default="default", metavar="ID", help="voice id (default: TTS_DEFAULT_VOICE)"
    )
    parser.add_argument(
        "--speed", type=float, metavar="FLOAT", help="0.5-2.0 (default: TTS_DEFAULT_SPEED)"
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="output file (default: <TTS_OUTPUT_DIR>/narration.<format>)",
    )
    parser.add_argument(
        "--format",
        choices=OUTPUT_FORMATS,
        help="audio format (default: the --output extension, else wav)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file instead of adding a _1, _2, ... suffix",
    )
    parser.add_argument(
        "--no-srt",
        action="store_true",
        help="do not write an .srt subtitle file next to the audio (TTS_SUBTITLES)",
    )
    parser.add_argument(
        "--keep-wav", action="store_true", help="with --format mp3, also keep a WAV copy"
    )
    parser.add_argument(
        "--pronunciations",
        metavar="PATH",
        help="pronunciation dictionary file, one 'word = respelling' per line "
        "(default: TTS_PRONUNCIATIONS)",
    )
    parser.add_argument(
        "--voice-sample",
        metavar="PATH",
        help="clone the voice in this audio clip (.wav/.flac/.mp3/.ogg, 5+ s of clean speech) "
        "with Chatterbox (default: TTS_VOICE_SAMPLE)",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        metavar="FLOAT",
        help="cloned voice expressiveness, 0.25-2.0 (default: TTS_EXAGGERATION, 0.5)",
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        metavar="FLOAT",
        help="cloned voice pacing, 0-1; lower is slower (default: TTS_CFG_WEIGHT, 0.5)",
    )
    parser.add_argument("--device", choices=DEVICES, help="override TTS_DEVICE")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose", "-v", action="store_true", help="debug logging and full tracebacks"
    )
    verbosity.add_argument("--quiet", "-q", action="store_true", help="only warnings and errors")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None, *, engine: TTSEngine | None = None) -> int:
    """Run the CLI and return its exit code. `engine` lets tests inject a mock engine."""
    _use_utf8_console()
    parser = build_parser()
    args_list = sys.argv[1:] if argv is None else argv
    if not args_list:
        parser.print_help()
        return EXIT_OK
    try:
        args = parser.parse_args(args_list)
        if not args.list_voices and args.text is None and args.file is None:
            parser.error("one of --text, --file or --list-voices is required")
    except SystemExit as exc:  # --help, --version, and usage errors
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE_ERROR

    try:
        settings = _apply_overrides(Settings.from_env(), args)
        level = "DEBUG" if args.verbose else "WARNING" if args.quiet else settings.log_level
        setup_logging(level)
        agent = TTSAgent(engine=engine, settings=settings)

        if args.list_voices:
            _print_voices(agent.list_voices(), agent.engine.default_voice, settings, sys.stdout)
            return EXIT_OK

        result = agent.generate(
            args.text,
            file=args.file,
            voice=args.voice,
            speed=args.speed,
            output_path=args.output,
            format=args.format,
            overwrite=args.overwrite,
            subtitles=False if args.no_srt else None,
            keep_wav=args.keep_wav,
            pronunciations=args.pronunciations,
        )
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception as exc:  # report every failure as one line, with a traceback if verbose
        return _report_error(exc, verbose=args.verbose)

    print(f"Saved:     {result.output_path}")
    if result.wav_path:
        print(f"WAV copy:  {result.wav_path}")
    if result.subtitle_path:
        print(f"Subtitles: {result.subtitle_path}")
    print(f"Duration:  {_format_duration(result.duration_seconds)}")
    sample = getattr(agent.engine, "voice_sample", None)
    print(f"Voice:     {f'cloned from {sample.name}' if sample else result.voice}")
    chunks = f"{result.chunk_count} chunk{'s' if result.chunk_count != 1 else ''}"
    load = f", plus {result.load_seconds:.1f} s loading" if result.load_seconds >= 0.1 else ""
    print(f"Generated: {result.generation_seconds:.1f} s ({chunks}{load})")
    if result.failed_chunks:
        numbers = ", ".join(str(i + 1) for i in result.failed_chunks)
        print(
            f"warning: {len(result.failed_chunks)} of {result.chunk_count} chunks failed and "
            f"were skipped: {numbers}",
            file=sys.stderr,
        )
        return EXIT_PARTIAL
    return EXIT_OK


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Apply the command-line options that override settings, and check they fit together."""
    overrides: dict[str, object] = {}
    if args.device:
        overrides["device"] = args.device
    if args.voice_sample:
        overrides["voice_sample"] = Path(args.voice_sample)
    if args.exaggeration is not None:
        overrides["exaggeration"] = args.exaggeration
    if args.cfg_weight is not None:
        overrides["cfg_weight"] = args.cfg_weight
    settings = dataclasses.replace(settings, **overrides) if overrides else settings
    if settings.voice_sample is None:
        if args.exaggeration is not None or args.cfg_weight is not None:
            raise InvalidOptionError(
                "--exaggeration and --cfg-weight only apply to voice cloning; "
                "add --voice-sample (or set TTS_VOICE_SAMPLE)"
            )
    elif args.voice.strip().lower() not in ("default", "clone"):
        raise InvalidOptionError(
            "--voice cannot be combined with voice cloning: the voice sample is the voice"
        )
    return settings


def _report_error(exc: Exception, *, verbose: bool) -> int:
    if verbose:
        traceback.print_exception(exc, file=sys.stderr)
    if isinstance(exc, TTSError):
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR if isinstance(exc, _USAGE_ERRORS) else EXIT_GENERATION_ERROR
    hint = "" if verbose else " (rerun with --verbose for details)"
    print(f"error: unexpected {type(exc).__name__}: {exc}{hint}", file=sys.stderr)
    return EXIT_GENERATION_ERROR


def _print_voices(
    voices: list[VoiceInfo], engine_default: str | None, settings: Settings, out: TextIO
) -> None:
    default_voice = engine_default or settings.default_voice
    rows = [("ID", "NAME", "ACCENT", "GENDER", "QUALITY")]
    rows += [(v.id, v.name, v.accent, v.gender, v.description) for v in voices]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        marker = "*" if row[0] == default_voice else " "
        line = "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        print(f"{marker} {line}".rstrip(), file=out)
    source = "" if engine_default else f" (TTS_DEFAULT_VOICE={default_voice})"
    print(f"\n* default voice{source}", file=out)


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}:{secs:04.1f} ({seconds:.1f} s)"


def _use_utf8_console() -> None:
    """The Windows console defaults to cp1252, which cannot print many characters."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")
