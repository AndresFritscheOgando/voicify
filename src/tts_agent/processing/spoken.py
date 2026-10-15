"""The text the engine reads, derived from the text the viewer sees.

Pronunciation entries and normalization rules rewrite parts of a chunk ("St." -> "Street",
"SQL" -> "sequel"). Subtitles must keep the original spelling, so every rewrite is recorded and
the engine's word timings are mapped back onto the original text.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tts_agent.engines.base import WordTiming


@dataclass(frozen=True)
class Rule:
    """Replace every match of `pattern` with `replace(match)`. Returning the match unchanged
    (or None) leaves it alone."""

    name: str
    pattern: re.Pattern[str]
    replace: Callable[[re.Match[str]], str | None]


@dataclass(frozen=True)
class Edit:
    spoken_start: int
    spoken_end: int
    display_start: int
    display_end: int


@dataclass(frozen=True)
class SpokenText:
    display: str
    text: str  # what the engine reads
    edits: tuple[Edit, ...] = ()

    def edit_at(self, spoken_offset: int) -> Edit | None:
        for edit in self.edits:
            if edit.spoken_start <= spoken_offset < edit.spoken_end:
                return edit
        return None


def to_spoken(display: str, rules: Sequence[Rule]) -> SpokenText:
    """Apply `rules` to `display` in one pass. Earlier rules win where matches overlap."""
    claimed: list[tuple[int, int, str]] = []
    for rule in rules:
        for match in rule.pattern.finditer(display):
            start, end = match.span()
            if start == end or any(start < e and s < end for s, e, _ in claimed):
                continue
            replacement = rule.replace(match)
            if replacement is None or replacement == match.group():
                continue
            claimed.append((start, end, replacement))
    if not claimed:
        return SpokenText(display, display)

    claimed.sort()
    parts: list[str] = []
    edits: list[Edit] = []
    position = 0
    length = 0
    for start, end, replacement in claimed:
        parts.append(display[position:start])
        length += start - position
        parts.append(replacement)
        edits.append(Edit(length, length + len(replacement), start, end))
        length += len(replacement)
        position = end
    parts.append(display[position:])
    return SpokenText(display, "".join(parts), tuple(edits))


def map_words(words: Sequence[WordTiming], spoken: SpokenText) -> list[WordTiming]:
    """Turn the engine's word timings (spelled as in `spoken.text`) into timings for the words
    of `spoken.display`. Tokens inside one rewrite merge into a single display word."""
    if not spoken.edits:
        return list(words)
    mapped: list[WordTiming] = []
    last_edit: Edit | None = None
    cursor = 0
    for word in words:
        if not word.text.strip():
            continue
        offset = spoken.text.find(word.text, cursor)
        if offset < 0:
            continue
        cursor = offset + len(word.text)
        edit = spoken.edit_at(offset)
        if edit is None:
            mapped.append(word)
            last_edit = None
        elif edit is last_edit and mapped:
            previous = mapped[-1]
            mapped[-1] = WordTiming(previous.text, previous.start, max(previous.end, word.end))
        else:
            text = spoken.display[edit.display_start : edit.display_end]
            mapped.append(WordTiming(text, word.start, word.end))
            last_edit = edit
    return mapped
