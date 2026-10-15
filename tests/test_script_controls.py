"""Phase 8: pause markers, pronunciation dictionary, number/abbreviation normalization."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from tts_agent import EmptyTextError, InputFileError, Settings, TTSAgent
from tts_agent.engines.base import WordTiming
from tts_agent.engines.mock_engine import FailingMockEngine, MockEngine
from tts_agent.processing.audio_processor import fade, leading_silence, trailing_silence
from tts_agent.processing.normalizer import NORMALIZATION_RULES
from tts_agent.processing.pronunciations import load_pronunciations
from tts_agent.processing.spoken import Rule, map_words, to_spoken
from tts_agent.processing.text_processor import chunk_text, parse_pause

RATE = 24_000


def normalize(text: str) -> str:
    return to_spoken(text, NORMALIZATION_RULES).text


def make_agent(tmp_path, engine=None, **settings) -> TTSAgent:
    options = {"output_dir": tmp_path, "sentence_pause_ms": 250, "paragraph_pause_ms": 700}
    options.update(settings)
    return TTSAgent(engine=engine or MockEngine(), settings=Settings(**options))


# --- pause markers: parsing ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("marker", "ms"),
    [
        ("[pause]", 1000),
        ("[pause 1.5s]", 1500),
        ("[pause 2s]", 2000),
        ("[pause 800ms]", 800),
        ("[pause 0.25 s]", 250),
        ("[PAUSE 3S]", 3000),
        ("[ pause  450ms ]", 450),
    ],
)
def test_parse_pause(marker, ms):
    assert parse_pause(marker) == ms


@pytest.mark.parametrize(
    "marker", ["[pause 2]", "[pause soon]", "[pause 1.5 seconds]", "[pause -1s]"]
)
def test_malformed_marker_warns_and_uses_one_second(marker, caplog):
    with caplog.at_level(logging.WARNING, logger="tts_agent"):
        assert parse_pause(marker) == 1000
    assert "malformed_pause_marker" in caplog.text


# --- pause markers: chunking -----------------------------------------------------------------


def test_marker_forces_a_chunk_boundary_and_is_removed():
    chunks = chunk_text("Hello there. [pause] How are you?", 400)

    assert [c.text for c in chunks] == ["Hello there.", "How are you?"]
    assert chunks[0].pause_after_ms == 1000
    assert chunks[1].pause_after_ms is None


def test_marker_inside_a_sentence():
    chunks = chunk_text("Wait for it [pause 800ms] now.", 400)

    assert [(c.text, c.pause_after_ms) for c in chunks] == [("Wait for it", 800), ("now.", None)]


def test_leading_and_trailing_markers():
    chunks = chunk_text("[pause 2s] Hello. [pause 500ms]", 400)

    assert len(chunks) == 1
    assert chunks[0].pause_before_ms == 2000
    assert chunks[0].pause_after_ms == 500


def test_consecutive_markers_add_up():
    chunks = chunk_text("One. [pause 1s][pause 500ms] [pause] Two.", 400)

    assert chunks[0].pause_after_ms == 2500


def test_marker_on_its_own_paragraph():
    chunks = chunk_text("First paragraph.\n\n[pause 3s]\n\nSecond paragraph.", 400)

    assert [c.text for c in chunks] == ["First paragraph.", "Second paragraph."]
    assert chunks[0].pause_after_ms == 3000
    assert chunks[0].ends_paragraph


def test_only_markers_is_empty_text():
    with pytest.raises(EmptyTextError):
        chunk_text("[pause] [pause 2s]\n\n[pause]", 400)


def test_markers_never_reach_chunk_text():
    chunks = chunk_text("A [pause] b [Pause 1s] c [PAUSE 300ms] d.", 400)

    assert all("[" not in c.text and "pause" not in c.text.lower() for c in chunks)


def test_brackets_that_are_not_markers_are_kept():
    chunks = chunk_text("He paused [briefly] and spoke.", 400)

    assert chunks[0].text == "He paused [briefly] and spoke."


# --- pause markers: audio --------------------------------------------------------------------


def gaps(result) -> list[float]:
    t = result.timeline
    return [b.start - a.end for a, b in zip(t, t[1:], strict=False)]


def test_marker_inserts_an_exact_pause(tmp_path):
    result = make_agent(tmp_path).generate("One. [pause 1.5s] Two. Three.\n\nFour.")

    # "Two. Three." is one chunk; markers split, sentences in a chunk do not
    assert gaps(result) == pytest.approx([1.5, 0.7])


def test_marker_replaces_the_paragraph_pause(tmp_path):
    result = make_agent(tmp_path).generate("One. [pause 300ms]\n\nTwo.")

    assert gaps(result) == pytest.approx([0.3])


def test_leading_and_trailing_pauses(tmp_path):
    result = make_agent(tmp_path).generate("[pause 2s] Hello there. [pause 1s]")

    assert result.timeline[0].start == pytest.approx(2.0)
    assert result.duration_seconds == pytest.approx(result.timeline[-1].end + 1.0)


def test_pause_survives_a_skipped_chunk(tmp_path):
    engine = FailingMockEngine(fail_on=("Broken",))
    result = make_agent(tmp_path, engine).generate("One. [pause 2s] Broken bit. [pause 1s] Three.")

    assert result.failed_chunks == [1]
    assert gaps(result) == pytest.approx([3.0])


class PaddedEngine(MockEngine):
    """Kokoro-like: speech with 200 ms of silence before and 300 ms after."""

    def synthesize_timed(self, text, voice, speed=1.0):
        speech, words = super().synthesize_timed(text, voice, speed)
        lead, tail = np.zeros(int(0.2 * RATE), np.float32), np.zeros(int(0.3 * RATE), np.float32)
        shifted = [WordTiming(w.text, w.start + 0.2, w.end + 0.2) for w in words]
        return np.concatenate([lead, speech, tail]), shifted


def audible_gaps(path) -> list[float]:
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32")
    loud = np.abs(audio) > 0.01
    edges = np.flatnonzero(np.diff(loud.astype(np.int8)))
    runs = [(edges[i] + 1, edges[i + 1] + 1) for i in range(0, len(edges) - 1)]
    return [(b - a) / rate for a, b in runs if not loud[a] and (b - a) / rate > 0.1]


def test_engine_silence_is_trimmed_around_markers(tmp_path):
    result = make_agent(tmp_path, PaddedEngine()).generate("One. [pause 1s] Two.")

    # audible silence between the two words is the marker alone, not 0.3 + 1.0 + 0.2
    gap = [g for g in audible_gaps(result.output_path) if g > 0.5]
    assert gap == pytest.approx([1.0], abs=0.02)


def test_engine_silence_is_kept_at_ordinary_pauses(tmp_path):
    result = make_agent(tmp_path, PaddedEngine()).generate("One.\n\nTwo.")

    gap = [g for g in audible_gaps(result.output_path) if g > 0.5]
    assert gap == pytest.approx([0.3 + 0.7 + 0.2], abs=0.02)


def test_words_follow_the_trim(tmp_path):
    result = make_agent(tmp_path, PaddedEngine()).generate("Hello one. [pause 1s] Hello two.")

    second = result.timeline[1]
    assert second.words[0].start == pytest.approx(second.start, abs=0.01)
    assert result.timeline[0].words[-1].end <= result.timeline[0].end


def test_subtitles_never_show_markers(tmp_path):
    result = make_agent(tmp_path).generate("Hello. [pause 2s] World. [pause] Again.")

    srt = result.subtitle_path.read_text(encoding="utf-8")
    assert "pause" not in srt.lower()
    assert "Hello." in srt and "World." in srt and "Again." in srt


# --- silence detection -----------------------------------------------------------------------


def test_leading_and_trailing_silence_detection():
    audio = np.concatenate([np.zeros(2400), 0.5 * np.ones(4800), np.zeros(1200)]).astype(np.float32)

    assert leading_silence(audio, RATE) == 2400
    assert trailing_silence(audio, RATE) == 1200


def test_all_silent_audio():
    audio = np.zeros(1000, dtype=np.float32)

    assert leading_silence(audio, RATE) == 1000 - 1000 % 120
    assert trailing_silence(audio, RATE) == 1000


def test_fade_ramps_the_cut_edge():
    audio = np.ones(1000, dtype=np.float32)

    faded_in, faded_out = fade(audio, RATE, fade_in=True), fade(audio, RATE, fade_in=False)

    assert faded_in[0] == 0 and faded_in[200] == 1
    assert faded_out[-1] == 0 and faded_out[0] == 1


# --- pronunciation dictionary ----------------------------------------------------------------


def write_dictionary(tmp_path, body: str):
    path = tmp_path / "pronunciations.txt"
    path.write_text(body, encoding="utf-8")
    return path


def test_dictionary_respells_whole_words_case_sensitively(tmp_path):
    rules = load_pronunciations(write_dictionary(tmp_path, "SQL = sequel\n"))

    assert to_spoken("SQL and SQLite and sql.", rules).text == "sequel and SQLite and sql."


def test_ignore_case_flag(tmp_path):
    rules = load_pronunciations(write_dictionary(tmp_path, "nginx/i = engine x\n"))

    assert to_spoken("NGINX, Nginx and nginx.", rules).text == ("engine x, engine x and engine x.")


def test_phonemes_in_slashes_use_misaki_inline_syntax(tmp_path):
    rules = load_pronunciations(write_dictionary(tmp_path, "Kokoro = /kˈOkəɹO/\n"))

    assert to_spoken("Say Kokoro now.", rules).text == "Say [Kokoro](/kˈOkəɹO/) now."


def test_multi_word_and_longest_entries_win(tmp_path):
    rules = load_pronunciations(
        write_dictionary(tmp_path, "Gull = Gul\nGull Point = Gull Pointe\n")
    )

    assert to_spoken("Gull Point and a Gull.", rules).text == "Gull Pointe and a Gul."


def test_keys_with_punctuation(tmp_path):
    rules = load_pronunciations(
        write_dictionary(tmp_path, "C++ = C plus plus\nDr. Who = Doctor Who\n")
    )

    assert to_spoken("Dr. Who uses C++.", rules).text == "Doctor Who uses C plus plus."


def test_comments_and_blank_lines_are_ignored(tmp_path):
    rules = load_pronunciations(write_dictionary(tmp_path, "# comment\n\n  SQL = sequel  \n"))

    assert len(rules) == 1


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("SQL sequel\n", ":1: expected 'word = pronunciation'"),
        ("# ok\nSQL =\n", ":2: both the word"),
        ("= sequel\n", ":1: both the word"),
        ("Kokoro = /kˈOkəɹO\n", ":1: phonemes must be written as /phonemes/"),
    ],
)
def test_dictionary_errors_name_the_line(tmp_path, body, message):
    with pytest.raises(InputFileError, match=message):
        load_pronunciations(write_dictionary(tmp_path, body))


def test_missing_dictionary(tmp_path):
    with pytest.raises(InputFileError, match="not found"):
        load_pronunciations(tmp_path / "nope.txt")


def test_duplicate_entry_warns_and_last_wins(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="tts_agent"):
        rules = load_pronunciations(
            write_dictionary(tmp_path, "SQL = sequel\nSQL = ess queue el\n")
        )

    assert "pronunciation_duplicate" in caplog.text
    assert to_spoken("SQL", rules).text == "ess queue el"


def test_engine_hears_the_dictionary_but_subtitles_do_not(tmp_path):
    path = write_dictionary(tmp_path, "SQL = sequel\n")
    engine = MockEngine()

    result = make_agent(tmp_path, engine).generate("I love SQL. It is great.", pronunciations=path)

    assert engine.calls[0][0] == "I love sequel. It is great."
    srt = result.subtitle_path.read_text(encoding="utf-8")
    assert "SQL" in srt and "sequel" not in srt


def test_dictionary_from_settings(tmp_path):
    path = write_dictionary(tmp_path, "SQL = sequel\n")
    engine = MockEngine()

    make_agent(tmp_path, engine, pronunciations=path).generate("SQL.")

    assert engine.calls[0][0] == "sequel."


def test_bad_dictionary_fails_before_synthesis(tmp_path):
    engine = MockEngine()

    with pytest.raises(InputFileError):
        make_agent(tmp_path, engine).generate("Hi.", pronunciations=tmp_path / "missing.txt")
    assert engine.calls == []


def test_dictionary_beats_normalization(tmp_path):
    path = write_dictionary(tmp_path, "St. Mary's = Saint Marys\n")
    engine = MockEngine()

    make_agent(tmp_path, engine).generate("We met at St. Mary's today.", pronunciations=path)

    assert engine.calls[0][0] == "We met at Saint Marys today."


# --- display/engine text mapping -------------------------------------------------------------


def test_to_spoken_without_matches_is_identity():
    spoken = to_spoken("Nothing to change.", NORMALIZATION_RULES)

    assert spoken.text == spoken.display and spoken.edits == ()


def test_earlier_rules_win_overlaps():
    import re

    first = Rule("first", re.compile(r"ab"), lambda m: "X")
    second = Rule("second", re.compile(r"bc"), lambda m: "Y")

    assert to_spoken("abc", [first, second]).text == "Xc"


def test_map_words_merges_tokens_of_one_rewrite():
    spoken = to_spoken("Call 555-1234 now.", NORMALIZATION_RULES)
    words = [
        WordTiming("Call", 0.0, 0.3),
        WordTiming("5", 0.3, 0.4),
        WordTiming("5", 0.4, 0.5),
        WordTiming("5", 0.5, 0.6),
        WordTiming("1", 0.7, 0.8),
        WordTiming("now", 1.5, 1.8),
    ]

    mapped = map_words(words, spoken)

    assert [(w.text, w.start, w.end) for w in mapped] == [
        ("Call", 0.0, 0.3),
        ("555-1234", 0.3, 0.8),
        ("now", 1.5, 1.8),
    ]


def test_subtitles_keep_original_spelling_with_normalization(tmp_path):
    result = make_agent(tmp_path).generate("Meet at 3:30 pm on Main St. now. Call 555-1234.")

    srt = result.subtitle_path.read_text(encoding="utf-8")
    assert "3:30 pm" in srt and "Main St." in srt and "555-1234" in srt


# --- normalization rules ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        # abbreviations misaki reads as letters or clipped words
        ("He lives on Main St. now.", "He lives on Main Street now."),
        ("He lives on Main St. Then he left.", "He lives on Main Street. Then he left."),
        ("We visited St. Louis.", "We visited Saint Louis."),
        ("Go to 5th Ave. today.", "Go to 5th Avenue today."),
        ("On Elm Rd. and Sunset Blvd. today.", "On Elm Road and Sunset Boulevard today."),
        ("Prof. Kim and Gov. Brown spoke.", "Professor Kim and Governor Brown spoke."),
        (
            "Gen. Lee, Capt. Hook, Lt. Dan, Sgt. Pepper.",
            "General Lee, Captain Hook, Lieutenant Dan, Sergeant Pepper.",
        ),
        ("Rev. King and Hon. Smith.", "Reverend King and Honorable Smith."),
        ("Up Mt. Everest near Ft. Worth.", "Up Mount Everest near Fort Worth."),
        ("John Smith Jr. spoke.", "John Smith Junior spoke."),
        ("John Smith, Sr., spoke.", "John Smith, Senior, spoke."),
        (
            "See Fig. 3, Vol. 2, Ch. 4, Sec. 5, Eq. 6.",
            "See Figure 3, Volume 2, Chapter 4, Section 5, Equation 6.",
        ),
        ("Read pp. 5 and Figs. 2.", "Read pages 5 and Figures 2."),
        (
            "The Dept. of Energy and the Univ. of Texas.",
            "The Department of Energy and the University of Texas.",
        ),
        ("Approx. 5 people.", "Approximately 5 people."),
        ("Acme Inc. Then more.", "Acme Incorporated. Then more."),
        ("Acme Corp. hired him.", "Acme Corporation hired him."),
        # units
        ("It is 5 km away.", "It is 5 kilometers away."),
        ("It weighs 1 kg.", "It weighs 1 kilogram."),
        ("Add 250 ml and 2 L.", "Add 250 milliliters and 2 liters."),
        ("It is 5 cm, 3 mm, 6 ft, 2 mi.", "It is 5 centimeters, 3 millimeters, 6 feet, 2 miles."),
        ("Going 60 mph or 90 km/h.", "Going 60 miles per hour or 90 kilometers per hour."),
        ("It was 70°F or 21 °C.", "It was 70 degrees Fahrenheit or 21 degrees Celsius."),
        ("A 5 GB file of 3 MB parts.", "A 5 gigabyte file of 3 megabytes parts."),
        ("Use 5 g, 10 mg, 2 lb and 4 oz.", "Use 5 grams, 10 milligrams, 2 pounds and 4 ounces."),
        # clock times
        ("At 3:30 pm.", "At 3 30 PM."),
        ("At 10:00 AM. Then", "At 10 AM. Then"),
        ("At 9:00.", "At 9 o'clock."),
        ("At 3:05 p.m. sharp.", "At 3 oh 5 PM sharp."),
        ("Meet at 3:05 p.m.", "Meet at 3 oh 5 PM."),
        ("At 14:30.", "At 14 30."),
        ("At 7am sharp.", "At 7 AM sharp."),
        # phone numbers
        ("Call (555) 123-4567.", "Call 5 5 5, 1 2 3, 4 5 6 7."),
        ("Call 555-123-4567.", "Call 5 5 5, 1 2 3, 4 5 6 7."),
        ("Call 1-800-555-1234.", "Call 1, 8 0 0, 5 5 5, 1 2 3 4."),
        ("Call 555-1234.", "Call 5 5 5, 1 2 3 4."),
        # ranges
        ("Items 1-10 here.", "Items 1 to 10 here."),
        ("Pages 10–20.", "Pages 10 to 20."),
        ("From 1990-2000 it grew.", "From 1990 to 2000 it grew."),
        ("Won 3-2, then lost.", "Won 3 to 2, then lost."),
        # Roman numerals
        ("World War II ended.", "World War 2 ended."),
        ("Chapter IV begins.", "Chapter 4 begins."),
        ("Henry VIII ruled.", "Henry the 8th ruled."),
        ("Louis XIV and Elizabeth II.", "Louis the 14th and Elizabeth the 2nd."),
        # fractions, multipliers, number signs
        ("Use 1/2 cup and 1 3/4 cups.", "Use one half cup and 1 and three quarters cups."),
        ("Add 1/3 or 2/3 or 1/4.", "Add one third or two thirds or one quarter."),
        ("It is 5x faster, even 2.5x.", "It is 5 times faster, even 2.5 times."),
        ("It was #1 in sales.", "It was number 1 in sales."),
    ],
)
def test_normalization_rule(text, spoken):
    assert normalize(text) == spoken


@pytest.mark.parametrize(
    "text",
    [
        # handled correctly by misaki already, or ambiguous: left alone
        "Dr. Smith paid $3.50 in 1999.",
        "Up 50% on May 4th, the 21st century.",
        "Pi is 3.14 and version 2.0 shipped.",
        "Mr. and Mrs. Jones met Ms. Park.",
        "The NASA and FBI agents used SQL.",
        "At 3 pm.",
        "I said I am.",
        "Size XL, type V, plan B.",
        "Rated 4/5 stars on 12/25/2024.",
        "At 1920x1080.",
        "On 2024-01-15 it rained.",
        "COVID-19 hit.",
        "Temps of -5 degrees.",
        "Is it 5 in long?",
        "Use v1.5-2 later.",
        "the st. john's wort",
        "Gov. is short for governor.",
        "She is no. 1 and Fig is a fruit.",
    ],
)
def test_normalization_leaves_text_alone(text):
    assert normalize(text) == text


def test_normalization_can_be_turned_off(tmp_path):
    engine = MockEngine()

    make_agent(tmp_path, engine, normalize=False).generate("Meet on Main St. at 3:30 pm.")

    assert engine.calls[0][0] == "Meet on Main St. at 3:30 pm."


def test_normalize_setting_from_environment():
    assert Settings.from_env({"TTS_NORMALIZE": "false"}).normalize is False
    assert Settings.from_env({}).normalize is True


# --- the rules really fix misaki (G2P only, no audio) ----------------------------------------


@pytest.mark.model
@pytest.mark.parametrize(
    ("text", "phonemes_before", "phonemes_after"),
    [
        ("Main St. now.", "sənt", "stɹˈit"),
        ("Go 5 km.", "kˌAˈɛm", "kəlˈɑmə"),
        ("At 10:00 AM.", "zˈiəɹO zˈiəɹO", "tˈɛn ˌAˈɛm"),
        ("Items 1-10.", "wˈʌntˌɛn", "wˈʌn tə tˈɛn"),
        ("Henry VIII ruled.", "vˌiˌIˌIˈI", "ˈAθ"),
    ],
)
def test_misaki_reads_the_normalized_text_better(text, phonemes_before, phonemes_after):
    import warnings

    warnings.filterwarnings("ignore")
    from kokoro import KPipeline

    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)

    def phonemes(t: str) -> str:
        return " ".join(r.phonemes for r in pipe(t))

    assert phonemes_before in phonemes(text)
    after = phonemes(normalize(text))
    assert phonemes_after in after
    assert phonemes_before not in after
