"""Saved chunks and `resume=True`: a failed or interrupted run picks up where it stopped."""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest
import soundfile as sf

from tts_agent import Settings, SynthesisError, TTSAgent
from tts_agent.cli import EXIT_GENERATION_ERROR, EXIT_OK, EXIT_PARTIAL, main
from tts_agent.engines.base import WordTiming
from tts_agent.engines.mock_engine import FailingMockEngine, MockEngine
from tts_agent.utils.part_cache import PartCache, parts_dir_for

SCRIPT = (
    "The first chunk has exactly one sentence here.\n\n"
    "The second chunk opens paragraph number two. "
    "The third chunk closes paragraph number two."
)
FIRST = "The first chunk has exactly one sentence here."
SECOND = "The second chunk opens paragraph number two."
THIRD = "The third chunk closes paragraph number two."


def make_agent(engine, tmp_path, **overrides) -> TTSAgent:
    options = {"output_dir": tmp_path / "outputs", "max_chunk_chars": 50}
    options.update(overrides)
    return TTSAgent(engine=engine, settings=Settings(**options))


def texts(engine: MockEngine) -> list[str]:
    return [text for text, _, _ in engine.calls]


def parts_dir(tmp_path):
    return tmp_path / "outputs" / ".narration.wav.parts"


class InterruptingEngine(MockEngine):
    """Raises KeyboardInterrupt (Ctrl+C) when it reaches a chosen text."""

    def __init__(self, stop_at: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.stop_at = stop_at

    def synthesize(self, text, voice, speed=1.0):
        if self.stop_at in text:
            raise KeyboardInterrupt
        return super().synthesize(text, voice, speed)


# --- saving and cleaning up ------------------------------------------------------------------


def test_successful_run_leaves_no_parts(tmp_path):
    make_agent(MockEngine(), tmp_path).generate(SCRIPT)

    assert not parts_dir(tmp_path).exists()
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == [
        "narration.srt",
        "narration.wav",
    ]


def test_parts_folder_sits_next_to_the_output(tmp_path):
    assert parts_dir_for(tmp_path / "ep" / "book.mp3") == tmp_path / "ep" / ".book.mp3.parts"


def test_aborted_run_keeps_the_finished_chunks(tmp_path):
    agent = make_agent(FailingMockEngine(fail_on=("third",)), tmp_path, on_chunk_error="abort")

    with pytest.raises(SynthesisError):
        agent.generate(SCRIPT)

    assert agent.kept_parts == parts_dir(tmp_path)
    assert len(list(parts_dir(tmp_path).glob("*.npz"))) == 2
    assert not list(parts_dir(tmp_path).glob("*.tmp"))


def test_run_where_nothing_finished_writes_nothing(tmp_path):
    agent = make_agent(FailingMockEngine(fail_on=("",)), tmp_path)

    with pytest.raises(SynthesisError, match="All 3"):
        agent.generate(SCRIPT)

    assert agent.kept_parts is None
    assert not (tmp_path / "outputs").exists()


def test_interrupted_run_keeps_the_finished_chunks(tmp_path, caplog):
    agent = make_agent(InterruptingEngine(stop_at="second"), tmp_path)

    with caplog.at_level(logging.WARNING, logger="tts_agent"), pytest.raises(KeyboardInterrupt):
        agent.generate(SCRIPT)

    assert agent.kept_parts == parts_dir(tmp_path)
    assert "parts_kept" in caplog.text
    assert "chunks=1/3" in caplog.text


# --- resuming --------------------------------------------------------------------------------


def test_resume_synthesizes_only_the_missing_chunks(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        make_agent(InterruptingEngine(stop_at="third"), tmp_path).generate(SCRIPT)

    engine = MockEngine()
    result = make_agent(engine, tmp_path).generate(SCRIPT, resume=True)

    assert texts(engine) == [THIRD]
    assert result.resumed_chunks == 2
    assert result.chunk_count == 3
    assert not parts_dir(tmp_path).exists()


def test_resumed_output_matches_a_clean_run(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        make_agent(InterruptingEngine(stop_at="third"), tmp_path).generate(SCRIPT)
    resumed = make_agent(MockEngine(), tmp_path).generate(SCRIPT, resume=True)
    clean = make_agent(MockEngine(), tmp_path / "clean").generate(SCRIPT)

    audio, _ = sf.read(resumed.output_path, dtype="float32")
    expected, _ = sf.read(clean.output_path, dtype="float32")
    np.testing.assert_array_equal(audio, expected)
    assert resumed.subtitle_path.read_text() == clean.subtitle_path.read_text()
    assert resumed.timeline == clean.timeline


def test_without_resume_saved_chunks_are_synthesized_again(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        make_agent(InterruptingEngine(stop_at="third"), tmp_path).generate(SCRIPT)

    engine = MockEngine()
    result = make_agent(engine, tmp_path).generate(SCRIPT)

    assert texts(engine) == [FIRST, SECOND, THIRD]
    assert result.resumed_chunks == 0
    assert not parts_dir(tmp_path).exists()


def test_skipped_chunks_keep_the_parts_and_resume_retries_only_them(tmp_path):
    failing = FailingMockEngine(fail_on=("second",))
    agent = make_agent(failing, tmp_path)
    first = agent.generate(SCRIPT)
    assert first.failed_chunks == [1]
    assert agent.kept_parts == parts_dir(tmp_path)

    engine = MockEngine()
    retried = make_agent(engine, tmp_path).generate(SCRIPT, resume=True, overwrite=True)

    assert texts(engine) == [SECOND]
    assert retried.failed_chunks == []
    assert retried.resumed_chunks == 2
    assert not parts_dir(tmp_path).exists()


def test_edited_sentence_is_the_only_one_synthesized_again(tmp_path):
    agent = make_agent(FailingMockEngine(fail_on=("third",)), tmp_path, on_chunk_error="abort")
    with pytest.raises(SynthesisError):
        agent.generate(SCRIPT)

    engine = MockEngine()
    edited = SCRIPT.replace("opens paragraph", "starts paragraph")
    make_agent(engine, tmp_path).generate(edited, resume=True)

    assert texts(engine) == [SECOND.replace("opens", "starts"), THIRD]


@pytest.mark.parametrize("change", [{"voice": "am_michael"}, {"speed": 1.5}])
def test_other_voice_or_speed_does_not_reuse_parts(tmp_path, change):
    agent = make_agent(FailingMockEngine(fail_on=("third",)), tmp_path, on_chunk_error="abort")
    with pytest.raises(SynthesisError):
        agent.generate(SCRIPT)

    engine = MockEngine()
    result = make_agent(engine, tmp_path).generate(SCRIPT, resume=True, **change)

    assert texts(engine) == [FIRST, SECOND, THIRD]
    assert result.resumed_chunks == 0


def test_other_engine_settings_do_not_reuse_parts(tmp_path):
    class OtherSettingsEngine(MockEngine):
        @property
        def cache_key(self) -> str:
            return "mock:louder"

    agent = make_agent(FailingMockEngine(fail_on=("third",)), tmp_path, on_chunk_error="abort")
    with pytest.raises(SynthesisError):
        agent.generate(SCRIPT)

    engine = OtherSettingsEngine()
    make_agent(engine, tmp_path).generate(SCRIPT, resume=True)

    assert len(engine.calls) == 3


def test_damaged_part_is_synthesized_again(tmp_path, caplog):
    agent = make_agent(FailingMockEngine(fail_on=("third",)), tmp_path, on_chunk_error="abort")
    with pytest.raises(SynthesisError):
        agent.generate(SCRIPT)
    for part in parts_dir(tmp_path).glob("*.npz"):
        part.write_bytes(b"not a part")

    engine = MockEngine()
    with caplog.at_level(logging.WARNING, logger="tts_agent"):
        result = make_agent(engine, tmp_path).generate(SCRIPT, resume=True)

    assert texts(engine) == [FIRST, SECOND, THIRD]
    assert result.resumed_chunks == 0
    assert "part_unreadable" in caplog.text


# --- the cache itself ------------------------------------------------------------------------


def test_part_round_trip_keeps_audio_and_word_timings(tmp_path):
    cache = PartCache(tmp_path / "parts", "mock", 24_000)
    audio = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
    words = [WordTiming("Hello", 0.0, 0.25), WordTiming("wörld", 0.3, 0.5)]

    cache.save("Hello wörld", "af_heart", 1.0, audio, words)
    loaded = cache.load("Hello wörld", "af_heart", 1.0)

    assert loaded is not None
    np.testing.assert_array_equal(loaded[0], audio)
    assert loaded[1] == words
    assert cache.saved == 1
    assert cache.count() == 1


def test_part_without_word_timings_round_trips(tmp_path):
    cache = PartCache(tmp_path / "parts", "mock", 24_000)
    cache.save("Hi.", "clone", 1.0, np.ones(10, dtype=np.float32), [])

    loaded = cache.load("Hi.", "clone", 1.0)

    assert loaded is not None
    assert loaded[1] == []


def test_missing_part_and_empty_cache(tmp_path):
    cache = PartCache(tmp_path / "parts", "mock", 24_000)

    assert cache.load("Hi.", "af_heart", 1.0) is None
    assert cache.count() == 0
    cache.clear()  # nothing to remove is fine


def test_key_depends_on_every_input(tmp_path):
    base = PartCache(tmp_path, "mock", 24_000).key("Hi.", "af_heart", 1.0)

    assert PartCache(tmp_path, "other", 24_000).key("Hi.", "af_heart", 1.0) != base
    assert PartCache(tmp_path, "mock", 22_050).key("Hi.", "af_heart", 1.0) != base
    assert PartCache(tmp_path, "mock", 24_000).key("Hi!", "af_heart", 1.0) != base
    assert PartCache(tmp_path, "mock", 24_000).key("Hi.", "am_michael", 1.0) != base
    assert PartCache(tmp_path, "mock", 24_000).key("Hi.", "af_heart", 1.1) != base


def test_part_that_cannot_be_saved_only_warns(tmp_path, caplog):
    blocker = tmp_path / "parts"
    blocker.write_text("a file where the folder should be")
    cache = PartCache(blocker, "mock", 24_000)

    with caplog.at_level(logging.WARNING, logger="tts_agent"):
        cache.save("Hi.", "af_heart", 1.0, np.ones(10, dtype=np.float32), [])

    assert cache.saved == 0
    assert "part_not_saved" in caplog.text


def test_unsaveable_parts_do_not_stop_generation(tmp_path):
    (tmp_path / "outputs").mkdir()
    parts_dir(tmp_path).write_text("blocks the parts folder")

    result = make_agent(MockEngine(), tmp_path).generate(SCRIPT)

    assert result.output_path.exists()


# --- command line ----------------------------------------------------------------------------


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in list(os.environ):
        if name.startswith("TTS_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("TTS_MAX_CHUNK_CHARS", "50")
    logger = logging.getLogger("tts_agent")
    saved = (logger.level, logger.propagate, list(logger.handlers))
    yield tmp_path
    logger.setLevel(saved[0])
    logger.propagate = saved[1]
    logger.handlers[:] = saved[2]


def test_cli_failure_points_to_resume(in_tmp, capsys, monkeypatch):
    monkeypatch.setenv("TTS_ON_CHUNK_ERROR", "abort")
    engine = FailingMockEngine(fail_on=("third",))

    assert main(["--text", SCRIPT, "--quiet"], engine=engine) == EXIT_GENERATION_ERROR

    stderr = capsys.readouterr().err
    assert "--resume to continue from where it stopped" in stderr
    assert ".narration.wav.parts" in stderr


def test_cli_interrupt_points_to_resume(in_tmp, capsys):
    engine = InterruptingEngine(stop_at="third")

    assert main(["--text", SCRIPT, "--quiet"], engine=engine) == 130

    assert "--resume to continue" in capsys.readouterr().err


def test_cli_partial_run_points_to_resume_and_resume_finishes_it(in_tmp, capsys):
    assert main(["--text", SCRIPT], engine=FailingMockEngine(fail_on=("second",))) == EXIT_PARTIAL
    assert "--resume to retry only the failed chunks" in capsys.readouterr().err

    engine = MockEngine()
    code = main(["--text", SCRIPT, "--resume", "--overwrite"], engine=engine)

    assert code == EXIT_OK
    assert texts(engine) == [SECOND]
    assert "Resumed:   2 of 3 chunks from saved parts" in capsys.readouterr().out


def test_cli_usage_error_prints_no_resume_hint(in_tmp, capsys):
    assert main(["--text", "Hi.", "--voice", "nobody"], engine=MockEngine()) != EXIT_OK

    assert "--resume" not in capsys.readouterr().err
