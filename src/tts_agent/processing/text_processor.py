"""Text preparation for long-form narration: normalize, split, and pack into chunks."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

from tts_agent.errors import EmptyTextError, InvalidOptionError

logger = logging.getLogger(__name__)

# Plain replacements. Em dashes are kept: misaki already reads "—" as a pause, while "--" loses
# the pause and turning a range dash ("10–20") into a hyphen makes misaki read "tentwenty".
_CHAR_MAP = str.maketrans(
    {
        "‘": "'",  # ‘
        "’": "'",  # ’
        "‚": "'",  # ‚
        "‛": "'",  # ‛
        "′": "'",  # ′
        "“": '"',  # “
        "”": '"',  # ”
        "„": '"',  # „
        "‟": '"',  # ‟
        "″": '"',  # ″
        "…": "...",  # …
        "―": "—",  # horizontal bar -> em dash
        " ": " ",  # no-break space
        " ": " ",  # thin space
        " ": " ",  # narrow no-break space
        "​": None,  # zero-width space
        "﻿": None,  # byte-order mark
        "­": None,  # soft hyphen
    }
)
_DOUBLE_HYPHEN = re.compile(r"\s*--+\s*")
_SPACED_EN_DASH = re.compile(r"\s+–\s+")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")
_WHITESPACE = re.compile(r"\s+")

# Words that end with a period but do not end a sentence (compared lowercase, without the dot).
# Abbreviations that are also everyday words ("No.", "Sun.", "Mar.") are left out on purpose:
# a missed split costs more than a spurious one.
ABBREVIATIONS = frozenset(
    """
    mr mrs ms mx dr prof sr jr st mt ft rev hon gen col capt lt sgt cmdr gov sen pres
    vs etc al e.g i.e cf viz approx dept univ inc ltd corp llc bros vol vols
    fig figs eq pp ch op jan feb apr jun jul aug sep sept oct nov dec
    tue tues thu thur thurs fri a.m p.m u.s u.k u.n e.u u.s.a d.c ph.d
    """.split()  # noqa: SIM905 - a word block is easier to scan than a 70-item list
)
# "etc." usually ends its sentence when the next word is capitalized, so it may split there.
_ABBREVIATIONS_THAT_MAY_END = frozenset({"etc"})

# A candidate boundary: terminal punctuation, optional closing quotes/brackets, whitespace.
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*(?=\s+)")
_NEXT_STARTS_SENTENCE = re.compile(r"\s+[\"'(\[]*[A-Z0-9]")
_LAST_WORD = re.compile(r"(\S+)$")

# Pause markers: [pause], [pause 1.5s], [pause 800ms] (case-insensitive).
PAUSE_MARKER = re.compile(r"\[\s*pause\b([^\]]*)\]", re.IGNORECASE)
_PAUSE_LENGTH = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s)", re.IGNORECASE)
DEFAULT_PAUSE_MS = 1000


@dataclass(frozen=True)
class Chunk:
    """A piece of text to synthesize in one engine call.

    `pause_before_ms` / `pause_after_ms` come from `[pause]` markers: an exact silence that
    replaces the usual sentence or paragraph pause at that boundary. None means no marker.
    """

    index: int
    text: str
    paragraph_index: int
    ends_paragraph: bool
    pause_before_ms: int | None = None
    pause_after_ms: int | None = None


def normalize_text(text: str) -> str:
    """NFC-normalize, replace typographic characters, and collapse whitespace.

    Paragraph breaks (blank lines) are kept as a single `\\n\\n`; other line breaks become spaces.
    """
    text = unicodedata.normalize("NFC", text).translate(_CHAR_MAP)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _DOUBLE_HYPHEN.sub(" — ", text)
    text = _SPACED_EN_DASH.sub(" — ", text)
    paragraphs = (_WHITESPACE.sub(" ", p).strip() for p in _PARAGRAPH_BREAK.split(text))
    return "\n\n".join(p for p in paragraphs if p)


def split_paragraphs(text: str) -> list[str]:
    """Split normalized text on blank lines."""
    return [p for p in text.split("\n\n") if p.strip()]


def split_sentences(paragraph: str) -> list[str]:
    """Split one paragraph into sentences.

    Handles abbreviations (`Dr.`, `e.g.`, `U.S.`), initials (`J. K.`), decimals (`3.14`) and
    ellipses. A boundary needs terminal punctuation followed by whitespace and a capital letter,
    digit, or opening quote/bracket.
    """
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(paragraph):
        end = match.end()
        if not _NEXT_STARTS_SENTENCE.match(paragraph, end):
            continue
        if _is_abbreviation(paragraph[start:end]):
            continue
        sentence = paragraph[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _is_abbreviation(candidate: str) -> bool:
    candidate = candidate.rstrip("\"')]")
    if not candidate.endswith(".") or candidate.endswith(".."):
        return False  # "!", "?", and ellipses always end a sentence
    match = _LAST_WORD.search(candidate)
    word = match.group(1).lstrip("\"'([").rstrip(".") if match else ""
    if word.lower() in _ABBREVIATIONS_THAT_MAY_END:
        return False
    if word.lower() in ABBREVIATIONS:
        return True
    return len(word) == 1 and word.isupper()  # an initial such as "J."


def parse_pause(marker: str) -> int:
    """Milliseconds for one `[pause ...]` marker. Malformed markers warn and use the default."""
    match = PAUSE_MARKER.fullmatch(marker.strip())
    argument = match.group(1).strip() if match else ""
    if not argument:
        return DEFAULT_PAUSE_MS
    length = _PAUSE_LENGTH.fullmatch(argument)
    if length is None:
        logger.warning(
            "malformed_pause_marker marker=%r using_ms=%d (write [pause 1.5s] or [pause 800ms])",
            marker,
            DEFAULT_PAUSE_MS,
        )
        return DEFAULT_PAUSE_MS
    value = float(length.group(1))
    return round(value if length.group(2).lower() == "ms" else value * 1000)


def strip_pause_markers(text: str) -> str:
    return PAUSE_MARKER.sub(" ", text)


def chunk_text(text: str, max_chars: int) -> list[Chunk]:
    """Normalize `text` and pack its sentences into chunks of at most `max_chars` characters.

    Chunks never cross a paragraph break and never split a sentence, unless that sentence alone
    is longer than `max_chars`; then it is split at `;`, `:`, `,`, and finally at whitespace.
    `[pause]` markers are removed from the text; each one forces a chunk boundary and records an
    exact pause on the chunk before it (or, at the very start, on the first chunk).
    """
    if max_chars < 1:
        raise InvalidOptionError(f"max_chars must be positive, got {max_chars}")
    normalized = normalize_text(text)
    if not strip_pause_markers(normalized).strip():
        raise EmptyTextError("Text is empty; nothing to synthesize")

    drafts: list[dict] = []
    leading_pause: int | None = None
    for paragraph_index, paragraph in enumerate(split_paragraphs(normalized)):
        paragraph_drafts: list[dict] = []
        position = 0
        for marker in [*PAUSE_MARKER.finditer(paragraph), None]:
            segment = paragraph[position : marker.start() if marker else len(paragraph)].strip()
            if segment:
                pieces: list[str] = []
                for sentence in split_sentences(segment):
                    pieces.extend(_split_long(sentence, max_chars))
                for piece in _pack(pieces, max_chars):
                    paragraph_drafts.append(
                        {"text": piece, "paragraph_index": paragraph_index, "pause_after": None}
                    )
            if marker is None:
                break
            pause = parse_pause(marker.group())
            previous = paragraph_drafts[-1] if paragraph_drafts else drafts[-1] if drafts else None
            if previous is None:
                leading_pause = (leading_pause or 0) + pause
            else:
                previous["pause_after"] = (previous["pause_after"] or 0) + pause
            position = marker.end()
        if paragraph_drafts:
            paragraph_drafts[-1]["ends_paragraph"] = True
        drafts.extend(paragraph_drafts)

    chunks = [
        Chunk(
            index=i,
            text=d["text"],
            paragraph_index=d["paragraph_index"],
            ends_paragraph=d.get("ends_paragraph", False),
            pause_before_ms=leading_pause if i == 0 else None,
            pause_after_ms=d["pause_after"],
        )
        for i, d in enumerate(drafts)
    ]
    check_integrity(strip_pause_markers(normalized), chunks)
    return chunks


def check_integrity(normalized: str, chunks: list[Chunk]) -> None:
    """Fail loudly if chunking dropped, duplicated, or reordered any text."""
    expected = _WHITESPACE.sub("", normalized)
    actual = _WHITESPACE.sub("", "".join(c.text for c in chunks))
    if actual != expected:
        raise AssertionError("Chunking integrity check failed: chunk text != normalized input")


def _pack(pieces: list[str], max_chars: int) -> list[str]:
    packed: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= max_chars:
            current = f"{current} {piece}"
        else:
            packed.append(current)
            current = piece
    if current:
        packed.append(current)
    return packed


_SPLIT_POINTS = (
    re.compile(r"(?<=;)\s+"),
    re.compile(r"(?<=:)\s+"),
    re.compile(r"(?<=,)\s+"),
    re.compile(r"\s+"),
)


def _split_long(sentence: str, max_chars: int, level: int = 0) -> list[str]:
    """Split a sentence longer than `max_chars` at the strongest available punctuation."""
    if len(sentence) <= max_chars:
        return [sentence]
    if level == len(_SPLIT_POINTS):
        # A single "word" longer than the limit: cut it into fixed-size pieces.
        return [sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)]
    parts = [p for p in _SPLIT_POINTS[level].split(sentence) if p]
    if len(parts) == 1:
        return _split_long(sentence, max_chars, level + 1)
    pieces: list[str] = []
    for part in parts:
        pieces.extend(_split_long(part, max_chars, level + 1))
    return _pack(pieces, max_chars)
