"""Pronunciation dictionary: per-project fixes for words the engine says wrong.

File format (UTF-8, one entry per line, `#` starts a comment line):

    Kokoro = Koh-koh-roh          respell it
    SQL = sequel
    nginx/i = engine x            `/i` after the word: match any capitalization
    Kokoro = /kˈOkəɹO/            slashes: exact phonemes (misaki's [word](/phonemes/) syntax)

Matching is whole-word and case-sensitive unless `/i` is given. Replacements change only what
the engine reads; subtitles keep the original spelling.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from tts_agent.errors import InputFileError
from tts_agent.processing.spoken import Rule

logger = logging.getLogger(__name__)

_IGNORE_CASE_FLAG = "/i"


def load_pronunciations(path: str | Path) -> list[Rule]:
    """Read a pronunciation dictionary. Raises `InputFileError` with the line number on errors."""
    path = Path(path)
    if not path.is_file():
        raise InputFileError(f"Pronunciation file not found: {path}")
    try:
        lines = path.read_bytes().decode("utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise InputFileError(f"Pronunciation file {path} is not valid UTF-8") from exc
    except OSError as exc:
        raise InputFileError(f"Could not read pronunciation file {path}: {exc}") from exc

    entries: dict[tuple[str, bool], str] = {}
    for number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise InputFileError(f"{path}:{number}: expected 'word = pronunciation', got {raw!r}")
        key, value = (part.strip() for part in line.split("=", 1))
        ignore_case = key.lower().endswith(_IGNORE_CASE_FLAG)
        if ignore_case:
            key = key[: -len(_IGNORE_CASE_FLAG)].strip()
        if not key or not value:
            raise InputFileError(f"{path}:{number}: both the word and its pronunciation are needed")
        if value.startswith("/") and (not value.endswith("/") or len(value) < 3):
            raise InputFileError(f"{path}:{number}: phonemes must be written as /phonemes/")
        if (key, ignore_case) in entries:
            logger.warning("pronunciation_duplicate file=%s line=%d word=%r", path, number, key)
        entries[(key, ignore_case)] = value

    # Longer words first, so "Gull Point" wins over "Gull".
    ordered = sorted(entries.items(), key=lambda item: len(item[0][0]), reverse=True)
    rules = [_rule(key, ignore_case, value) for (key, ignore_case), value in ordered]
    logger.info("pronunciations_loaded file=%s entries=%d", path, len(rules))
    return rules


def _rule(key: str, ignore_case: bool, value: str) -> Rule:
    pattern = re.compile(rf"(?<!\w){re.escape(key)}(?!\w)", re.IGNORECASE if ignore_case else 0)
    if value.startswith("/"):
        return Rule(f"pronunciation:{key}", pattern, lambda m: f"[{m.group()}]({value})")
    return Rule(f"pronunciation:{key}", pattern, lambda m: value)
