from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from tts_agent.engines.base import WordTiming
from tts_agent.processing.subtitles import (
    MAX_CUE_SECONDS,
    Cue,
    build_cues,
    format_srt,
    wrap_lines,
    write_srt,
)

SRT_BLOCK = re.compile(
    r"(\d+)\n(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})\n(.+?)(?:\n\n|\n?$)",
    re.S,
)


@dataclass(frozen=True)
class Chunk:
    text: str
    start: float
    end: float
    words: tuple[WordTiming, ...] = ()


def timed(text: str, start: float, seconds_per_char: float = 0.05) -> Chunk:
    """A chunk whose words are timed evenly by character position, like MockEngine."""
    words = tuple(
        WordTiming(
            m.group(), start + m.start() * seconds_per_char, start + m.end() * seconds_per_char
        )
        for m in re.finditer(r"\w+(?:'\w+)*|[^\w\s]", text)
    )
    return Chunk(text, start, start + len(text) * seconds_per_char, words)


def parse_srt(text: str) -> list[tuple[int, float, float, str]]:
    cues = []
    for m in SRT_BLOCK.finditer(text):
        g = [int(x) for x in m.groups()[:9]]
        start = g[1] * 3600 + g[2] * 60 + g[3] + g[4] / 1000
        end = g[5] * 3600 + g[6] * 60 + g[7] + g[8] / 1000
        cues.append((g[0], start, end, m.group(10)))
    return cues


def one_line(cues: list[Cue]) -> str:
    return " ".join(c.text.replace("\n", " ") for c in cues)


# --- wrapping and formatting -----------------------------------------------------------------


def test_short_text_is_not_wrapped():
    assert wrap_lines("Hello there.") == "Hello there."


def test_long_text_wraps_into_two_balanced_lines():
    text = "This sentence is a bit too long to fit on one subtitle line."

    lines = wrap_lines(text).split("\n")

    assert len(lines) == 2
    assert all(len(line) <= 42 for line in lines)
    assert abs(len(lines[0]) - len(lines[1])) <= 12
    assert " ".join(lines) == text


def test_text_without_spaces_is_left_alone():
    assert wrap_lines("x" * 60) == "x" * 60


def test_format_srt():
    srt = format_srt([Cue(0.0, 1.5, "Hello."), Cue(3661.0, 3662.25, "One\nTwo")])

    assert srt == (
        "1\n00:00:00,000 --> 00:00:01,500\nHello.\n\n2\n01:01:01,000 --> 01:01:02,250\nOne\nTwo\n"
    )


def test_write_srt_round_trips(tmp_path):
    cues = [Cue(0.2, 1.0, "First."), Cue(1.2, 2.0, "Second\nline.")]

    path = write_srt(cues, tmp_path / "a.srt")

    assert parse_srt(path.read_text(encoding="utf-8")) == [
        (1, 0.2, 1.0, "First."),
        (2, 1.2, 2.0, "Second\nline."),
    ]


# --- cue timing ------------------------------------------------------------------------------


def test_one_cue_per_sentence_timed_from_words():
    chunk = timed("Hello there. How are you?", start=10.0)

    cues = build_cues([chunk], audio_duration=20.0)

    assert [c.text for c in cues] == ["Hello there.", "How are you?"]
    assert cues[0].start == pytest.approx(10.0)
    # "How" starts at character 13
    assert cues[1].start == pytest.approx(10.0 + 13 * 0.05)
    assert cues[1].end == pytest.approx(chunk.end)


def test_word_timings_beat_character_proportions():
    # The first sentence is spoken slowly: its words take far longer than its share of characters.
    text = "Slow. Fast words here."
    words = (
        WordTiming("Slow", 0.0, 3.0),
        WordTiming(".", 3.0, 3.0),
        WordTiming("Fast", 3.2, 3.5),
        WordTiming("words", 3.5, 3.8),
        WordTiming("here", 3.8, 4.0),
        WordTiming(".", 4.0, 4.0),
    )
    cues = build_cues([Chunk(text, 0.0, 4.0, words)], audio_duration=4.0)

    assert cues[1].start == pytest.approx(3.2)  # proportional would say about 1.1 s


def test_proportional_fallback_without_word_timings():
    chunk = Chunk("Aaaa. Bbbbbbbbb.", start=2.0, end=4.0)  # 16 characters over 2 seconds

    cues = build_cues([chunk], audio_duration=4.0)

    assert [c.text for c in cues] == ["Aaaa.", "Bbbbbbbbb."]
    assert cues[0].start == pytest.approx(2.0)
    assert cues[1].start == pytest.approx(2.0 + 2.0 * 6 / 16)
    assert cues[1].end == pytest.approx(4.0)


def test_unknown_tokens_are_skipped():
    words = (
        WordTiming("Hello", 0.0, 0.5),
        WordTiming("hɛloʊ", 0.5, 0.6),  # an engine spelling that is not in the text
        WordTiming("world", 0.7, 1.2),
    )

    cues = build_cues([Chunk("Hello world.", 0.0, 1.3, words)], audio_duration=1.3)

    assert cues == [Cue(0.0, 1.2, "Hello world.")]


def test_cues_across_chunks_keep_the_timeline():
    chunks = [timed("First chunk.", 0.0), timed("Second chunk.", 5.0)]

    cues = build_cues(chunks, audio_duration=6.0)

    assert [c.text for c in cues] == ["First chunk.", "Second chunk."]
    assert cues[1].start == pytest.approx(5.0)


def test_long_sentence_is_split_to_fit_two_lines():
    sentence = (
        "This is a single long sentence that goes on and on without stopping for breath "
        "and would never fit into a two line subtitle of forty two characters per line."
    )
    cues = build_cues([timed(sentence, 0.0, 0.01)], audio_duration=10.0)

    assert len(cues) >= 2
    for cue in cues:
        lines = cue.text.split("\n")
        assert len(lines) <= 2
        assert len(cue.text.replace("\n", " ")) <= 84
    assert one_line(cues) == sentence


def test_split_prefers_clause_boundaries():
    sentence = (
        "When the storm finally passed over the harbor, the whole town came down to the water "
        "to see what was left."
    )
    cues = build_cues([timed(sentence, 0.0, 0.01)], audio_duration=10.0)

    assert cues[0].text.replace("\n", " ").endswith("harbor,")


def test_split_is_balanced_without_a_stranded_tail():
    sentence = (
        "It stands on a narrow spit of granite at the far end of a fishing town "
        "that most maps forget to name."
    )  # 101 characters: too long for one cue, so two cues of similar length
    cues = build_cues([timed(sentence, 0.0, 0.01)], audio_duration=5.0)

    lengths = [len(c.text.replace("\n", " ")) for c in cues]
    assert len(lengths) == 2
    assert min(lengths) >= 40


def test_slow_sentence_is_split_by_duration():
    sentence = "Every word in this sentence is spoken very very slowly indeed."
    cues = build_cues([timed(sentence, 0.0, 0.3)], audio_duration=30.0)  # about 18 seconds

    assert len(cues) >= 3
    assert all(c.end - c.start <= MAX_CUE_SECONDS + 1e-9 for c in cues)
    assert one_line(cues) == sentence


def test_unvoiced_sentence_gets_a_short_cue_when_there_is_room():
    words = (WordTiming("One", 0.0, 0.5), WordTiming("Three", 1.0, 1.5))
    chunk = Chunk("One. Two. Three.", 0.0, 1.6, words)

    cues = build_cues([chunk], audio_duration=1.6)

    assert [c.text for c in cues] == ["One.", "Two.", "Three."]
    assert cues[1].start == pytest.approx(0.5)
    assert cues[1].end == pytest.approx(0.8)


def test_unvoiced_sentence_without_room_joins_the_previous_cue():
    words = (WordTiming("One", 0.0, 0.5), WordTiming("Three", 0.5, 1.0))
    chunk = Chunk("One. Two. Three.", 0.0, 1.1, words)

    cues = build_cues([chunk], audio_duration=1.1)

    assert [c.text for c in cues] == ["One. Two.", "Three."]


def test_cues_are_ordered_non_overlapping_and_inside_the_audio():
    chunks = [
        timed(f"Sentence {i} is here. And another one follows it.", i * 3.0) for i in range(5)
    ]
    duration = chunks[-1].end - 0.2  # pretend the audio was trimmed a little

    cues = build_cues(chunks, audio_duration=duration)

    for a, b in zip(cues, cues[1:], strict=False):
        assert a.end <= b.start
    assert all(c.start < c.end for c in cues)
    assert cues[-1].end <= duration


def test_very_short_cues_are_stretched_when_there_is_room():
    words = (WordTiming("Hi", 0.0, 0.05),)
    cues = build_cues([Chunk("Hi.", 0.0, 0.05, words)], audio_duration=2.0)

    assert cues[0].end - cues[0].start == pytest.approx(0.3)


def test_no_chunks_means_no_cues():
    assert build_cues([], audio_duration=1.0) == []


def test_write_srt_reports_unwritable_path(tmp_path):
    from tts_agent.errors import OutputError

    with pytest.raises(OutputError, match="Could not write subtitle"):
        write_srt([Cue(0.0, 1.0, "Hi.")], tmp_path)  # a directory


def test_single_overlong_word_gets_its_own_cue():
    word = "x" * 100
    cues = build_cues([timed(f"Start {word} end.", 0.0, 0.01)], audio_duration=5.0)

    assert word in [c.text for c in cues]
    assert one_line(cues) == f"Start {word} end."


def test_unvoiced_first_sentence_joins_the_next_cue():
    words = (WordTiming("Two", 0.0, 0.5),)
    chunk = Chunk("One. Two.", 0.0, 0.6, words)

    cues = build_cues([chunk], audio_duration=0.6)

    assert one_line(cues) == "One. Two."
    assert len(cues) == 1


def test_blank_tokens_are_ignored():
    words = (WordTiming(" ", 0.0, 0.1), WordTiming("Hi", 0.1, 0.4))

    assert build_cues([Chunk("Hi.", 0.0, 0.5, words)], audio_duration=0.5)[0].start == 0.1
