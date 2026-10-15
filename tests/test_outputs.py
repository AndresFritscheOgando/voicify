"""MP3 output, kept WAVs, and SRT subtitles, end to end through TTSAgent with a mock engine."""

from __future__ import annotations

import re
import sys

import numpy as np
import pytest
import soundfile as sf

from tts_agent import InvalidOptionError, OutputError, Settings, TTSAgent
from tts_agent.engines.mock_engine import FailingMockEngine, MockEngine
from tts_agent.processing.text_processor import normalize_text

SCRIPT = (
    "Welcome to our channel. Today we are going to explore how a small model can narrate "
    "an entire video script, one sentence at a time.\n\n"
    "Let's get started. First, we will look at the lighthouse at Gull Point."
)
SRT_TIME = re.compile(r"(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})")


def make_agent(tmp_path, engine=None, **settings) -> TTSAgent:
    options = {"output_dir": tmp_path, "max_chunk_chars": 80}
    options.update(settings)
    return TTSAgent(engine=engine or MockEngine(), settings=Settings(**options))


def read_srt(path) -> list[tuple[float, float, str]]:
    cues = []
    for block in path.read_text(encoding="utf-8").strip().split("\n\n"):
        lines = block.split("\n")
        g = [int(x) for x in SRT_TIME.fullmatch(lines[1]).groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        cues.append((start, end, " ".join(lines[2:])))
    return cues


# --- MP3 -------------------------------------------------------------------------------------


def test_mp3_output_matches_the_wav_duration(tmp_path):
    agent = make_agent(tmp_path)

    mp3 = agent.generate(SCRIPT, output_path=tmp_path / "a.mp3")
    wav = agent.generate(SCRIPT, output_path=tmp_path / "a.wav")

    assert mp3.format == "mp3"
    assert sf.info(mp3.output_path).format == "MP3"
    mp3_seconds = sf.info(mp3.output_path).duration
    assert abs(mp3_seconds - sf.info(wav.output_path).duration) < 0.05
    assert mp3_seconds == pytest.approx(mp3.duration_seconds, abs=1e-3)


def test_format_mp3_without_extension(tmp_path):
    result = make_agent(tmp_path).generate(SCRIPT, output_path=tmp_path / "clip", format="mp3")

    assert result.output_path == tmp_path / "clip.mp3"


def test_default_mp3_name(tmp_path):
    result = make_agent(tmp_path).generate(SCRIPT, format="mp3")

    assert result.output_path == tmp_path / "narration.mp3"
    assert result.subtitle_path == tmp_path / "narration.srt"


def test_mp3_does_not_keep_a_wav_by_default(tmp_path):
    result = make_agent(tmp_path).generate(SCRIPT, format="mp3")

    assert result.wav_path is None
    assert not (tmp_path / "narration.wav").exists()


def test_keep_wav_writes_a_matching_wav(tmp_path):
    result = make_agent(tmp_path).generate(SCRIPT, format="mp3", keep_wav=True)

    assert result.wav_path == tmp_path / "narration.wav"
    assert sf.info(result.wav_path).frames == round(result.duration_seconds * 24_000)


def test_keep_wav_requires_mp3(tmp_path):
    with pytest.raises(InvalidOptionError, match="keep_wav"):
        make_agent(tmp_path).generate(SCRIPT, keep_wav=True)


def test_missing_lameenc_fails_clearly_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "lameenc", None)

    with pytest.raises(OutputError, match="lameenc"):
        make_agent(tmp_path).generate(SCRIPT, format="mp3")
    assert list(tmp_path.iterdir()) == []


# --- subtitles -------------------------------------------------------------------------------


def test_srt_is_written_next_to_the_audio(tmp_path):
    result = make_agent(tmp_path).generate(SCRIPT, output_path=tmp_path / "ep1.wav")

    assert result.subtitle_path == tmp_path / "ep1.srt"
    assert result.subtitle_path.exists()


def test_srt_covers_every_word_in_order(tmp_path):
    result = make_agent(tmp_path).generate(SCRIPT)

    cues = read_srt(result.subtitle_path)
    spoken = " ".join(text for _, _, text in cues)
    assert spoken.split() == normalize_text(SCRIPT).split()


def test_srt_is_monotonic_and_inside_the_audio(tmp_path):
    result = make_agent(tmp_path).generate(SCRIPT)

    cues = read_srt(result.subtitle_path)
    assert len(cues) >= 5  # one per sentence at least
    for (s1, e1, _), (s2, _, _) in zip(cues, cues[1:], strict=False):
        assert s1 < e1 <= s2
    assert cues[-1][1] <= result.duration_seconds + 0.0005


def test_srt_lines_fit_two_lines_of_about_42_characters(tmp_path):
    long_sentence = " ".join(["narration"] * 40) + "."
    result = make_agent(tmp_path, max_chunk_chars=400).generate(long_sentence)

    blocks = result.subtitle_path.read_text(encoding="utf-8").strip().split("\n\n")
    assert len(blocks) > 1
    for block in blocks:
        text_lines = block.split("\n")[2:]
        assert 1 <= len(text_lines) <= 2
        assert all(len(line) <= 42 for line in text_lines)


def test_srt_cues_follow_the_paragraph_pause(tmp_path):
    agent = make_agent(tmp_path, paragraph_pause_ms=2000)
    result = agent.generate(SCRIPT)

    cues = read_srt(result.subtitle_path)
    second_paragraph = next(c for c in cues if c[2].startswith("Let's get started."))
    first_paragraph_end = max(e for s, e, t in cues if s < second_paragraph[0])
    assert second_paragraph[0] - first_paragraph_end >= 1.9


def test_subtitles_can_be_turned_off(tmp_path):
    result = make_agent(tmp_path).generate(SCRIPT, subtitles=False)

    assert result.subtitle_path is None
    assert not (tmp_path / "narration.srt").exists()


def test_subtitles_setting_is_the_default(tmp_path):
    result = make_agent(tmp_path, subtitles=False).generate(SCRIPT)
    assert result.subtitle_path is None

    result = make_agent(tmp_path, subtitles=False).generate(SCRIPT, subtitles=True)
    assert result.subtitle_path is not None


def test_engine_without_word_timings_still_gets_subtitles(tmp_path):
    result = make_agent(tmp_path, MockEngine(word_timings=False)).generate(SCRIPT)

    cues = read_srt(result.subtitle_path)
    assert " ".join(t for _, _, t in cues).split() == normalize_text(SCRIPT).split()


def test_skipped_chunks_have_no_cues(tmp_path):
    engine = FailingMockEngine(fail_on=("lighthouse",))
    result = make_agent(tmp_path, engine).generate(SCRIPT)

    assert result.failed_chunks
    text = result.subtitle_path.read_text(encoding="utf-8")
    assert "lighthouse" not in text
    assert "Welcome to our channel." in text


def test_existing_srt_moves_the_whole_set_to_a_free_name(tmp_path):
    (tmp_path / "narration.srt").write_text("keep me", encoding="utf-8")

    result = make_agent(tmp_path).generate(SCRIPT)

    assert result.output_path == tmp_path / "narration_1.wav"
    assert result.subtitle_path == tmp_path / "narration_1.srt"
    assert (tmp_path / "narration.srt").read_text(encoding="utf-8") == "keep me"
    assert not (tmp_path / "narration.wav").exists()


def test_existing_wav_is_respected_when_keeping_one(tmp_path):
    (tmp_path / "narration.wav").write_bytes(b"keep me")

    result = make_agent(tmp_path).generate(SCRIPT, format="mp3", keep_wav=True)

    assert result.output_path == tmp_path / "narration_1.mp3"
    assert result.wav_path == tmp_path / "narration_1.wav"
    assert (tmp_path / "narration.wav").read_bytes() == b"keep me"


def test_overwrite_replaces_audio_and_subtitles(tmp_path):
    (tmp_path / "narration.wav").write_bytes(b"old")
    (tmp_path / "narration.srt").write_text("old", encoding="utf-8")

    result = make_agent(tmp_path).generate(SCRIPT, overwrite=True)

    assert result.output_path == tmp_path / "narration.wav"
    assert (tmp_path / "narration.srt").read_text(encoding="utf-8") != "old"


def test_timeline_words_use_the_output_clock(tmp_path):
    result = make_agent(tmp_path, sentence_pause_ms=500).generate(SCRIPT)

    for timing in result.timeline:
        assert timing.words
        assert timing.words[0].start >= timing.start - 1e-9
        assert timing.words[-1].end <= timing.end + 1e-9
    starts = [w.start for t in result.timeline for w in t.words]
    assert starts == sorted(starts)
    assert np.isclose(result.timeline[-1].end, result.duration_seconds)
