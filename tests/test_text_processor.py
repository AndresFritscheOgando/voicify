from __future__ import annotations

import re

import pytest

from tts_agent.errors import EmptyTextError, InvalidOptionError
from tts_agent.processing.text_processor import (
    Chunk,
    check_integrity,
    chunk_text,
    normalize_text,
    split_paragraphs,
    split_sentences,
)


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


# --- normalization ---------------------------------------------------------------------------


def test_smart_quotes_become_plain():
    assert normalize_text("“Hi,” she said. It’s ‘fine’.") == ("\"Hi,\" she said. It's 'fine'.")


def test_ellipsis_character_becomes_three_dots():
    assert normalize_text("Wait… what?") == "Wait... what?"


@pytest.mark.parametrize("text", ["late--very late", "late -- very late", "late – very late"])
def test_double_hyphen_and_spaced_en_dash_become_em_dash(text):
    assert normalize_text(text) == "late — very late"


def test_em_dash_and_range_en_dash_are_kept():
    assert normalize_text("late—very late") == "late—very late"
    assert normalize_text("pages 10–20") == "pages 10–20"


def test_unicode_is_nfc_normalized():
    assert normalize_text("Café") == "Café"


def test_whitespace_collapses_but_paragraphs_survive():
    text = "  One\tline\nwraps   here.\r\n\r\n\n  Second para.  \n \n\nThird. "
    assert normalize_text(text) == "One line wraps here.\n\nSecond para.\n\nThird."


def test_invisible_characters_are_removed():
    assert normalize_text("﻿zero​width soft­hyphen") == "zerowidth softhyphen"


def test_blank_text_normalizes_to_empty():
    assert normalize_text(" \n\n \t ") == ""


def test_split_paragraphs():
    assert split_paragraphs("A. B.\n\nC.") == ["A. B.", "C."]


# --- sentence splitting ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two! Three? Four.", ["One.", "Two!", "Three?", "Four."]),
        ("Dr. Smith arrived. He sat.", ["Dr. Smith arrived.", "He sat."]),
        ("Mr. and Mrs. Jones left. Done.", ["Mr. and Mrs. Jones left.", "Done."]),
        ("Use tools, e.g. Hammers. Then rest.", ["Use tools, e.g. Hammers.", "Then rest."]),
        ("That is, i.e. Nothing. Fine.", ["That is, i.e. Nothing.", "Fine."]),
        ("She moved to the U.S. In May. Ok.", ["She moved to the U.S. In May.", "Ok."]),
        ("It costs 3.50 today. Pi is 3.14159.", ["It costs 3.50 today.", "Pi is 3.14159."]),
        ("J. K. Rowling wrote it. Yes.", ["J. K. Rowling wrote it.", "Yes."]),
        ("Wait... what? Really.", ["Wait... what?", "Really."]),
        ("He paused... Then he spoke.", ["He paused...", "Then he spoke."]),
        ('"Stop!" she said. He did.', ['"Stop!" she said.', "He did."]),
        ('She said "Go." Then left.', ['She said "Go."', "Then left."]),
        ("Is it (really) done? (Yes.) Good.", ["Is it (really) done?", "(Yes.)", "Good."]),
        ("Bring pens, paper, etc. Then start.", ["Bring pens, paper, etc.", "Then start."]),
        ("Chapter 1. 2010 was a year.", ["Chapter 1.", "2010 was a year."]),
        ("I said no. Then I left.", ["I said no.", "Then I left."]),
        ("Visit example.com today. Bye.", ["Visit example.com today.", "Bye."]),
        ("Version 2. it was fine.", ["Version 2. it was fine."]),
        ("no terminal punctuation", ["no terminal punctuation"]),
    ],
)
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


# --- chunking --------------------------------------------------------------------------------


def make_script(sentences: int, per_paragraph: int = 7) -> str:
    paragraphs = []
    for p in range(0, sentences, per_paragraph):
        paragraphs.append(
            " ".join(
                f"Sentence number {i} talks about item {i * 3} in some detail."
                for i in range(p, min(p + per_paragraph, sentences))
            )
        )
    return "\n\n".join(paragraphs)


def test_short_text_is_one_chunk():
    assert chunk_text("Hello there. How are you?", 400) == [
        Chunk(index=0, text="Hello there. How are you?", paragraph_index=0, ends_paragraph=True)
    ]


def test_empty_text_raises():
    with pytest.raises(EmptyTextError):
        chunk_text(" \n\n ", 400)


def test_non_positive_limit_raises():
    with pytest.raises(InvalidOptionError):
        chunk_text("Hi.", 0)


@pytest.mark.parametrize("max_chars", [60, 120, 400])
def test_chunks_respect_the_limit(max_chars):
    chunks = chunk_text(make_script(200), max_chars)

    assert all(len(c.text) <= max_chars for c in chunks)


def test_chunks_are_packed_greedily():
    chunks = chunk_text("Aaaa. Bbbb. Cccc. Dddd. Eeee.", 12)

    assert [c.text for c in chunks] == ["Aaaa. Bbbb.", "Cccc. Dddd.", "Eeee."]


def test_chunks_keep_sentences_whole():
    script = make_script(120)
    sentences = {s for p in split_paragraphs(normalize_text(script)) for s in split_sentences(p)}

    for chunk in chunk_text(script, 150):
        # every chunk is a run of whole sentences
        assert all(s in sentences for s in split_sentences(chunk.text))


def test_order_is_preserved_without_duplicates_or_omissions():
    script = make_script(500)
    chunks = chunk_text(script, 400)

    assert [c.index for c in chunks] == list(range(len(chunks)))
    numbers = [int(n) for c in chunks for n in re.findall(r"Sentence number (\d+)", c.text)]
    assert numbers == list(range(500))
    assert squash(" ".join(c.text for c in chunks)) == squash(normalize_text(script))


def test_chunks_do_not_cross_paragraphs():
    chunks = chunk_text("First para one. First para two.\n\nSecond para.\n\nThird.", 400)

    assert [(c.text, c.paragraph_index, c.ends_paragraph) for c in chunks] == [
        ("First para one. First para two.", 0, True),
        ("Second para.", 1, True),
        ("Third.", 2, True),
    ]


def test_only_the_last_chunk_of_a_paragraph_ends_it():
    chunks = chunk_text(make_script(14), 120)

    by_paragraph: dict[int, list[Chunk]] = {}
    for c in chunks:
        by_paragraph.setdefault(c.paragraph_index, []).append(c)
    assert list(by_paragraph) == [0, 1]
    for group in by_paragraph.values():
        assert len(group) > 1
        assert [c.ends_paragraph for c in group] == [False] * (len(group) - 1) + [True]


def test_oversized_sentence_splits_at_semicolons_first():
    sentence = "alpha beta gamma; delta epsilon zeta; eta theta iota, kappa lambda."

    chunks = chunk_text(sentence, 40)

    assert [c.text for c in chunks] == [
        "alpha beta gamma; delta epsilon zeta;",
        "eta theta iota, kappa lambda.",
    ]


def test_oversized_sentence_splits_at_colon_then_comma():
    sentence = "Here is the list: apples and pears, plums and cherries, figs and dates."

    chunks = chunk_text(sentence, 40)

    assert [c.text for c in chunks] == [
        "Here is the list:",
        "apples and pears, plums and cherries,",
        "figs and dates.",
    ]


def test_oversized_sentence_without_punctuation_splits_at_whitespace():
    sentence = " ".join(f"word{i}" for i in range(40)) + "."

    chunks = chunk_text(sentence, 50)

    assert len(chunks) > 1
    assert all(len(c.text) <= 50 for c in chunks)
    assert " ".join(c.text for c in chunks) == sentence


def test_single_huge_word_is_cut_hard():
    chunks = chunk_text("x" * 130, 50)

    assert [len(c.text) for c in chunks] == [50, 50, 30]


def test_oversized_sentence_stays_within_limit_next_to_short_ones():
    long = "a, " * 40 + "end."
    chunks = chunk_text(f"Short one. {long} Short two.", 50)

    assert all(len(c.text) <= 50 for c in chunks)
    assert chunks[0].text.startswith("Short one.")
    assert chunks[-1].text.endswith("Short two.")


def test_integrity_check_detects_missing_text():
    with pytest.raises(AssertionError):
        check_integrity("One. Two.", [Chunk(0, "One.", 0, True)])


def test_integrity_check_detects_duplicates():
    with pytest.raises(AssertionError):
        check_integrity("One.", [Chunk(0, "One.", 0, False), Chunk(1, "One.", 0, True)])
