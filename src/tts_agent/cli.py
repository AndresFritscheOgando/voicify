"""Command-line interface. Generation options arrive in Phase 3 and Phase 6."""

from __future__ import annotations

import argparse

from tts_agent import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voicify", description="Local text-to-speech agent.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0
