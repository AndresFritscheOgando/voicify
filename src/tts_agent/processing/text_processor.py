"""Text preparation for long-form narration: normalize, split, and pack into chunks."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from tts_agent.errors import EmptyTextError, InvalidOptionError

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


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    paragraph_index: int
    ends_paragraph: bool


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


def chunk_text(text: str, max_chars: int) -> list[Chunk]:
    """Normalize `text` and pack its sentences into chunks of at most `max_chars` characters.

    Chunks never cross a paragraph break and never split a sentence, unless that sentence alone
    is longer than `max_chars`; then it is split at `;`, `:`, `,`, and finally at whitespace.
    """
    if max_chars < 1:
        raise InvalidOptionError(f"max_chars must be positive, got {max_chars}")
    normalized = normalize_text(text)
    if not normalized:
        raise EmptyTextError("Text is empty; nothing to synthesize")

    chunks: list[Chunk] = []
    for paragraph_index, paragraph in enumerate(split_paragraphs(normalized)):
        pieces: list[str] = []
        for sentence in split_sentences(paragraph):
            pieces.extend(_split_long(sentence, max_chars))
        packed = _pack(pieces, max_chars)
        for i, chunk in enumerate(packed):
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=chunk,
                    paragraph_index=paragraph_index,
                    ends_paragraph=i == len(packed) - 1,
                )
            )
    check_integrity(normalized, chunks)
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
