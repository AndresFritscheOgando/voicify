"""Turn the chunk timeline into sentence-level SRT subtitles."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tts_agent.engines.base import WordTiming
from tts_agent.errors import OutputError
from tts_agent.processing.text_processor import split_sentences

MAX_LINE_CHARS = 42
MAX_LINES = 2
MAX_CUE_SECONDS = 7.0
MIN_CUE_SECONDS = 0.3
_BREAK_AFTER = re.compile(r"[,;:—]$")
_WORD = re.compile(r"\S+")


class TimedChunk(Protocol):
    """What subtitles need from one synthesized chunk (`ChunkTiming` fits). Times are absolute."""

    @property
    def text(self) -> str: ...
    @property
    def start(self) -> float: ...
    @property
    def end(self) -> float: ...
    @property
    def words(self) -> Sequence[WordTiming]: ...


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str  # may contain one line break


@dataclass(frozen=True)
class _Word:
    text: str
    start: float
    end: float


def build_cues(
    chunks: Sequence[TimedChunk],
    audio_duration: float,
    max_line_chars: int = MAX_LINE_CHARS,
    max_cue_seconds: float = MAX_CUE_SECONDS,
) -> list[Cue]:
    """One cue per sentence, split further when it would not fit two lines or runs too long.

    Sentence times come from the engine's word timings when it provides them, otherwise from
    splitting each chunk's span in proportion to character count.
    """
    max_chars = max_line_chars * MAX_LINES
    cues: list[Cue] = []
    for chunk in chunks:
        words = _time_words(chunk)
        for sentence in _group_by_sentence(chunk.text, words):
            for group in _split_words(sentence, max_chars, max_cue_seconds):
                cues.append(
                    Cue(
                        start=group[0].start,
                        end=group[-1].end,
                        text=wrap_lines(" ".join(w.text for w in group), max_line_chars),
                    )
                )
    return _tidy(cues, audio_duration, max_line_chars)


def wrap_lines(text: str, max_line_chars: int = MAX_LINE_CHARS) -> str:
    """Break `text` into at most two balanced lines at a space."""
    if len(text) <= max_line_chars:
        return text
    spaces = [i for i, c in enumerate(text) if c == " "]
    if not spaces:
        return text
    middle = len(text) / 2
    fitting = [i for i in spaces if i <= max_line_chars and len(text) - i - 1 <= max_line_chars]
    split = min(fitting or spaces, key=lambda i: abs(i - middle))
    return f"{text[:split]}\n{text[split + 1 :]}"


def format_srt(cues: Sequence[Cue]) -> str:
    blocks = [
        f"{n}\n{_timestamp(cue.start)} --> {_timestamp(cue.end)}\n{cue.text}\n"
        for n, cue in enumerate(cues, start=1)
    ]
    return "\n".join(blocks)


def write_srt(cues: Sequence[Cue], path: Path) -> Path:
    path = Path(path)
    try:
        path.write_text(format_srt(cues), encoding="utf-8", newline="\n")
    except OSError as exc:
        raise OutputError(f"Could not write subtitle file {path}: {exc}") from exc
    return path


def _timestamp(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _time_words(chunk: TimedChunk) -> list[tuple[int, _Word]]:
    """Every whitespace-separated word of the chunk text with its char offset and times."""
    spans = [(m.start(), m.end(), m.group()) for m in _WORD.finditer(chunk.text)]
    located = _locate_tokens(chunk)
    if not located:
        return _proportional_words(chunk, spans)

    timed: list[tuple[int, _Word]] = []
    previous_end = chunk.start
    for start, end, text in spans:
        times = [(s, e) for offset, s, e in located if start <= offset < end]
        if times:
            word = _Word(text, min(s for s, _ in times), max(e for _, e in times))
        else:  # punctuation-only "words" such as a lone dash are not voiced
            word = _Word(text, previous_end, previous_end)
        previous_end = word.end
        timed.append((start, word))
    return timed


def _locate_tokens(chunk: TimedChunk) -> list[tuple[int, float, float]]:
    """Find each timed token in the chunk text, in order. Tokens that do not appear are skipped."""
    located = []
    cursor = 0
    for token in chunk.words:
        if not token.text.strip():
            continue
        offset = chunk.text.find(token.text, cursor)
        if offset < 0:
            continue  # the engine spelled it differently; its neighbours still anchor the timing
        located.append((offset, token.start, token.end))
        cursor = offset + len(token.text)
    return located


def _proportional_words(
    chunk: TimedChunk, spans: list[tuple[int, int, str]]
) -> list[tuple[int, _Word]]:
    total = max(len(chunk.text), 1)
    span = chunk.end - chunk.start
    return [
        (start, _Word(text, chunk.start + span * start / total, chunk.start + span * end / total))
        for start, end, text in spans
    ]


def _group_by_sentence(text: str, words: list[tuple[int, _Word]]) -> list[list[_Word]]:
    bounds = []
    cursor = 0
    for sentence in split_sentences(text):
        start = text.find(sentence, cursor)
        cursor = start + len(sentence)
        bounds.append(cursor)
    groups: list[list[_Word]] = [[] for _ in bounds]
    index = 0
    for offset, word in words:
        while index < len(bounds) - 1 and offset >= bounds[index]:
            index += 1
        groups[index].append(word)
    return [g for g in groups if g]


def _split_words(words: list[_Word], max_chars: int, max_seconds: float) -> list[list[_Word]]:
    """Split a sentence into cues within the character and duration limits.

    The sentence is cut into as many cues as the limits require, aiming for equal lengths so
    no cue is left with a stranded tail. A cue past half its target also ends after a word that
    closes a clause, which reads more naturally.
    """
    n = len(words)
    prefix = [0]
    for word in words:
        prefix.append(prefix[-1] + len(word.text))

    def chars(i: int, j: int) -> int:  # characters of words[i:j] joined by spaces
        return prefix[j] - prefix[i] + (j - i - 1)

    total = chars(0, n)
    duration = words[-1].end - words[0].start
    pieces = max(math.ceil(total / max_chars), math.ceil(duration / max_seconds), 1)
    if pieces == 1:
        return [words]

    best: tuple[float, list[int]] | None = None
    for count, extra_cost in ((pieces, 0.0), (pieces + 1, _EXTRA_CUE_COST)):
        found = _best_breaks(words, count, chars, max_chars, max_seconds)
        if found and (best is None or found[0] + extra_cost < best[0]):
            best = (found[0] + extra_cost, found[1])
    if best is None:  # a single word breaks the limits; fall back to one word per cue
        return [[w] for w in words]
    cuts = [0, *best[1], n]
    return [words[a:b] for a, b in zip(cuts, cuts[1:], strict=False)]


_CLAUSE_BONUS = 0.15  # how much a break after , ; : or a dash is worth versus even lengths
_EXTRA_CUE_COST = 0.3


def _best_breaks(
    words: list[_Word],
    count: int,
    chars: Callable[[int, int], int],
    max_chars: int,
    max_seconds: float,
) -> tuple[float, list[int]] | None:
    """Split `words` into exactly `count` cues, minimizing squared length imbalance minus a
    bonus for clause breaks. Returns (cost, break indices) or None if no split fits."""
    n = len(words)
    target = chars(0, n) / count
    inf = float("inf")
    # cost[c][j]: best cost of splitting words[:j] into c cues; back[c][j]: where cue c starts
    cost = [[inf] * (n + 1) for _ in range(count + 1)]
    back = [[0] * (n + 1) for _ in range(count + 1)]
    cost[0][0] = 0.0
    for c in range(1, count + 1):
        for j in range(c, n + 1):
            for i in range(c - 1, j):
                if cost[c - 1][i] == inf:
                    continue
                length = chars(i, j)
                if length > max_chars or (
                    j - i > 1 and words[j - 1].end - words[i].start > max_seconds
                ):
                    continue
                piece = ((length - target) / target) ** 2
                if j < n and _BREAK_AFTER.search(words[j - 1].text):
                    piece -= _CLAUSE_BONUS
                if cost[c - 1][i] + piece < cost[c][j]:
                    cost[c][j] = cost[c - 1][i] + piece
                    back[c][j] = i
    if cost[count][n] == inf:
        return None
    breaks, j = [], n
    for c in range(count, 0, -1):
        j = back[c][j]
        breaks.append(j)
    return cost[count][n], sorted(b for b in breaks if b > 0)


def _tidy(cues: list[Cue], audio_duration: float, max_line_chars: int) -> list[Cue]:
    """Make cues ordered, non-overlapping, at least MIN_CUE_SECONDS where room allows, and
    within the audio. A cue left with no time (unvoiced punctuation) joins a neighbour."""
    tidy: list[Cue] = []
    carry = ""  # text of a zero-length cue that came before any real cue
    for i, cue in enumerate(cues):
        text = cue.text
        if carry:
            text = wrap_lines(f"{carry} {_one_line(text)}", max_line_chars)
            carry = ""
        start = min(max(cue.start, tidy[-1].end if tidy else 0.0), audio_duration)
        next_start = cues[i + 1].start if i + 1 < len(cues) else audio_duration
        room = min(max(next_start, cue.end), audio_duration)
        end = min(max(cue.end, start + MIN_CUE_SECONDS), room)
        if end > start:
            tidy.append(Cue(start, end, text))
        elif tidy:
            last = tidy[-1]
            joined = f"{_one_line(last.text)} {_one_line(text)}"
            tidy[-1] = Cue(last.start, last.end, wrap_lines(joined, max_line_chars))
        else:
            carry = _one_line(text)
    return tidy


def _one_line(text: str) -> str:
    return text.replace("\n", " ")
