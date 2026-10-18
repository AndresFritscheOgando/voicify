"""Takes with an implausible length (a cloned voice rambling or cutting off) are retried."""

from __future__ import annotations

import logging
import os

import pytest

from tts_agent import Settings, SynthesisError, TTSAgent
from tts_agent.cli import EXIT_OK, main
from tts_agent.engines.base import TTSEngine
from tts_agent.engines.mock_engine import MockEngine

SCRIPT = (
    "The first chunk has exactly one sentence here.\n\n"
    "The second chunk opens paragraph number two. "
    "The third chunk closes paragraph number two."
)
SECOND = "The second chunk opens paragraph number two."
RATE = 24_000
SECONDS_PER_CHAR = 0.01  # MockEngine's default


def expected(text: str) -> float:
    return len(text) * SECONDS_PER_CHAR


class LengthEngine(MockEngine):
    """A MockEngine that believes 0.5x-2x its normal length, and whose takes of texts containing
    `marker` come out `stretches[n]` times as long on the n-th try (then normal)."""

    def __init__(self, stretches=(), marker: str = "", fail_on_try: int | None = None) -> None:
        super().__init__()
        self.stretches = list(stretches)
        self.marker = marker
        self.fail_on_try = fail_on_try
        self.tries = 0

    def plausible_seconds(self, text, speed=1.0):
        return expected(text) * 0.5 / speed, expected(text) * 2.0 / speed

    def synthesize(self, text, voice, speed=1.0):
        audio = super().synthesize(text, voice, speed)
        if self.marker not in text:
            return audio
        self.tries += 1
        if self.tries == self.fail_on_try:
            raise RuntimeError("mock failure")
        stretch = self.stretches[self.tries - 1] if self.tries <= len(self.stretches) else 1.0
        return MockEngine.synthesize(
            MockEngine(seconds_per_char=SECONDS_PER_CHAR * stretch), text, voice, speed
        )


def make_agent(engine, tmp_path, **overrides) -> TTSAgent:
    options = {"output_dir": tmp_path / "outputs", "max_chunk_chars": 50}
    options.update(overrides)
    return TTSAgent(engine=engine, settings=Settings(**options))


def chunk_seconds(result, index: int) -> float:
    timing = next(t for t in result.timeline if t.index == index)
    return timing.end - timing.start


def test_engines_accept_any_length_by_default():
    assert TTSEngine.plausible_seconds(MockEngine(), "Hello there.", 1.0) is None


def test_too_long_take_is_synthesized_again(tmp_path, caplog):
    engine = LengthEngine(stretches=[5.0], marker="second")

    with caplog.at_level(logging.WARNING, logger="tts_agent"):
        result = make_agent(engine, tmp_path).generate(SCRIPT)

    assert engine.tries == 2
    assert result.suspect_chunks == []
    assert result.failed_chunks == []
    assert chunk_seconds(result, 1) == pytest.approx(expected(SECOND), abs=0.01)
    assert "chunk_implausible_length chunk=2/3 attempt=1/2" in caplog.text


def test_too_short_take_is_synthesized_again(tmp_path):
    engine = LengthEngine(stretches=[0.1], marker="second")

    result = make_agent(engine, tmp_path).generate(SCRIPT)

    assert engine.tries == 2
    assert result.suspect_chunks == []
    assert chunk_seconds(result, 1) == pytest.approx(expected(SECOND), abs=0.01)


def test_retry_count_applies_to_implausible_takes(tmp_path):
    engine = LengthEngine(stretches=[5.0, 5.0, 5.0], marker="second")

    result = make_agent(engine, tmp_path, chunk_retries=3).generate(SCRIPT)

    assert engine.tries == 4
    assert result.suspect_chunks == []


def test_when_every_take_is_implausible_the_closest_is_kept(tmp_path, caplog):
    engine = LengthEngine(stretches=[6.0, 3.0], marker="second")

    with caplog.at_level(logging.WARNING, logger="tts_agent"):
        result = make_agent(engine, tmp_path).generate(SCRIPT)

    assert result.suspect_chunks == [1]
    assert result.failed_chunks == []  # the sentence is kept, not dropped
    assert chunk_seconds(result, 1) == pytest.approx(expected(SECOND) * 3.0, abs=0.01)
    assert "chunk_length_kept chunk=2/3" in caplog.text


def test_closest_take_compares_short_and_long_by_ratio(tmp_path):
    # 0.1x is 5 times below the 0.5x floor; 6x is only 3 times above the 2x ceiling.
    engine = LengthEngine(stretches=[0.1, 6.0], marker="second")

    result = make_agent(engine, tmp_path).generate(SCRIPT)

    assert chunk_seconds(result, 1) == pytest.approx(expected(SECOND) * 6.0, abs=0.01)


@pytest.mark.parametrize("fail_on_try", [1, 2])
def test_implausible_take_beats_an_error(tmp_path, fail_on_try):
    engine = LengthEngine(stretches=[5.0, 5.0], marker="second", fail_on_try=fail_on_try)

    result = make_agent(engine, tmp_path).generate(SCRIPT)

    assert result.failed_chunks == []
    assert result.suspect_chunks == [1]


def test_abort_policy_keeps_an_implausible_take(tmp_path):
    engine = LengthEngine(stretches=[5.0, 5.0], marker="second")

    result = make_agent(engine, tmp_path, on_chunk_error="abort").generate(SCRIPT)

    assert result.suspect_chunks == [1]


def test_implausible_takes_are_not_saved_for_resume(tmp_path):
    class AlsoFailsThird(LengthEngine):
        def synthesize(self, text, voice, speed=1.0):
            if "third" in text:
                raise RuntimeError("mock failure")
            return super().synthesize(text, voice, speed)

    agent = make_agent(AlsoFailsThird(stretches=[5.0, 5.0], marker="second"), tmp_path)
    first = agent.generate(SCRIPT)
    assert first.failed_chunks == [2]
    assert first.suspect_chunks == [1]

    engine = MockEngine()
    resumed = make_agent(engine, tmp_path).generate(SCRIPT, resume=True, overwrite=True)

    assert [text for text, _, _ in engine.calls] == [
        SECOND,
        "The third chunk closes paragraph number two.",
    ]
    assert resumed.resumed_chunks == 1


def test_every_attempt_raising_still_fails_the_chunk(tmp_path):
    class Broken(LengthEngine):
        def synthesize(self, text, voice, speed=1.0):
            raise RuntimeError("mock failure")

    with pytest.raises(SynthesisError, match="All 3"):
        make_agent(Broken(), tmp_path).generate(SCRIPT)


def test_cli_warns_about_suspect_chunks(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for name in list(os.environ):
        if name.startswith("TTS_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("TTS_MAX_CHUNK_CHARS", "50")
    logger = logging.getLogger("tts_agent")
    saved = (logger.level, logger.propagate, list(logger.handlers))
    try:
        code = main(["--text", SCRIPT, "--quiet"], engine=LengthEngine([5.0, 5.0], marker="second"))
    finally:
        logger.setLevel(saved[0])
        logger.propagate = saved[1]
        logger.handlers[:] = saved[2]

    assert code == EXIT_OK  # the audio is complete; the warning asks for a listen
    err = capsys.readouterr().err
    assert "warning: chunk 2 came out much shorter or longer" in err
    assert "TTS_CHUNK_RETRIES" in err
