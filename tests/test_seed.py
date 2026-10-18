"""Seeds make the takes of a randomly sampling engine (a cloned voice) repeatable."""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest
import soundfile as sf

from tts_agent import InvalidOptionError, Settings, TTSAgent
from tts_agent.agent import take_seed
from tts_agent.cli import EXIT_OK, EXIT_USAGE_ERROR, main
from tts_agent.config import MAX_SEED
from tts_agent.engines.mock_engine import MockEngine

SCRIPT = (
    "The first chunk has exactly one sentence here.\n\n"
    "The second chunk opens paragraph number two. "
    "The third chunk closes paragraph number two."
)
FIRST = "The first chunk has exactly one sentence here."


class RandomEngine(MockEngine):
    """Like Chatterbox: every take is random unless seeded. Records each seed it is given."""

    def __init__(self) -> None:
        super().__init__()
        self.seeds: list[int] = []
        self._rng = np.random.default_rng()

    @property
    def uses_seed(self) -> bool:
        return True

    def seed(self, value: int) -> None:
        self.seeds.append(value)
        self._rng = np.random.default_rng(value)

    def synthesize(self, text, voice, speed=1.0):
        audio = super().synthesize(text, voice, speed)
        return (audio * self._rng.uniform(0.5, 1.0)).astype(np.float32)


def make_agent(engine, tmp_path, **overrides) -> TTSAgent:
    options = {"output_dir": tmp_path / "outputs", "max_chunk_chars": 50}
    options.update(overrides)
    return TTSAgent(engine=engine, settings=Settings(**options))


def audio_of(result) -> np.ndarray:
    return sf.read(result.output_path, dtype="float32")[0]


# --- settings --------------------------------------------------------------------------------


def test_seed_comes_from_the_environment():
    assert Settings.from_env({"TTS_SEED": "1234"}).seed == 1234
    assert Settings.from_env({}).seed is None


@pytest.mark.parametrize("value", ["-1", "1.5", "abc", str(MAX_SEED + 1)])
def test_bad_seed_in_the_environment_is_rejected(value):
    with pytest.raises(InvalidOptionError, match="TTS_SEED"):
        Settings.from_env({"TTS_SEED": value})


@pytest.mark.parametrize("value", [-1, MAX_SEED + 1, 1.5, True])
def test_bad_seed_argument_is_rejected(tmp_path, value):
    with pytest.raises(InvalidOptionError, match="seed"):
        make_agent(RandomEngine(), tmp_path).generate(SCRIPT, seed=value)


def test_take_seed_is_stable_and_fits_every_rng():
    assert take_seed(7, "Hello.", 1) == take_seed(7, "Hello.", 1)
    assert take_seed(7, "Hello.", 1) != take_seed(8, "Hello.", 1)
    assert take_seed(7, "Hello.", 1) != take_seed(7, "Hello!", 1)
    assert take_seed(7, "Hello.", 1) != take_seed(7, "Hello.", 2)
    assert 0 <= take_seed(MAX_SEED, "x" * 1000, 99) <= MAX_SEED


# --- repeatable takes ------------------------------------------------------------------------


def test_same_seed_gives_the_same_audio(tmp_path):
    a = make_agent(RandomEngine(), tmp_path / "a").generate(SCRIPT, seed=42)
    b = make_agent(RandomEngine(), tmp_path / "b").generate(SCRIPT, seed=42)

    np.testing.assert_array_equal(audio_of(a), audio_of(b))
    assert a.seed == b.seed == 42


def test_different_seeds_give_different_audio(tmp_path):
    a = make_agent(RandomEngine(), tmp_path / "a").generate(SCRIPT, seed=1)
    b = make_agent(RandomEngine(), tmp_path / "b").generate(SCRIPT, seed=2)

    assert not np.array_equal(audio_of(a), audio_of(b))


def test_without_a_seed_a_random_one_is_reported_and_repeats_the_run(tmp_path):
    first = make_agent(RandomEngine(), tmp_path / "a").generate(SCRIPT)
    again = make_agent(RandomEngine(), tmp_path / "b").generate(SCRIPT, seed=first.seed)

    assert first.seed is not None and 0 <= first.seed <= MAX_SEED
    np.testing.assert_array_equal(audio_of(first), audio_of(again))


def test_random_seeds_differ_between_runs(tmp_path):
    seeds = {make_agent(RandomEngine(), tmp_path / str(i)).generate(FIRST).seed for i in range(5)}

    assert len(seeds) > 1


def test_seed_setting_is_the_default_and_the_argument_wins(tmp_path):
    engine = RandomEngine()
    agent = make_agent(engine, tmp_path, seed=5)

    assert agent.generate(FIRST).seed == 5
    assert agent.generate(FIRST, seed=6, overwrite=True).seed == 6


def test_each_take_is_seeded_from_its_text(tmp_path):
    engine = RandomEngine()

    make_agent(engine, tmp_path).generate(SCRIPT, seed=9)

    assert len(engine.calls) == 3
    assert engine.seeds == [take_seed(9, text, 1) for text, _, _ in engine.calls]


def test_editing_one_sentence_keeps_the_other_takes(tmp_path):
    original = RandomEngine()
    make_agent(original, tmp_path / "a").generate(SCRIPT, seed=3)
    edited = RandomEngine()
    make_agent(edited, tmp_path / "b").generate(
        "A new opening sentence goes first.\n\n" + SCRIPT, seed=3
    )

    # The three original sentences get the same seeds, now one position later.
    assert edited.seeds[1:] == original.seeds


def test_a_retry_gets_a_new_seed(tmp_path):
    class FailsOnce(RandomEngine):
        def synthesize(self, text, voice, speed=1.0):
            if "first" in text and len(self.seeds) == 1:
                raise RuntimeError("mock failure")
            return super().synthesize(text, voice, speed)

    engine = FailsOnce()
    make_agent(engine, tmp_path).generate(FIRST, seed=4)

    assert engine.seeds == [take_seed(4, FIRST, 1), take_seed(4, FIRST, 2)]


def test_resumed_run_with_the_same_seed_matches_a_clean_one(tmp_path):
    class StopsAtThird(RandomEngine):
        def synthesize(self, text, voice, speed=1.0):
            if "third" in text:
                raise KeyboardInterrupt
            return super().synthesize(text, voice, speed)

    with pytest.raises(KeyboardInterrupt):
        make_agent(StopsAtThird(), tmp_path).generate(SCRIPT, seed=11)
    resumed = make_agent(RandomEngine(), tmp_path).generate(SCRIPT, seed=11, resume=True)
    clean = make_agent(RandomEngine(), tmp_path / "clean").generate(SCRIPT, seed=11)

    assert resumed.resumed_chunks == 2
    np.testing.assert_array_equal(audio_of(resumed), audio_of(clean))


def test_deterministic_engines_are_never_seeded(tmp_path):
    class Watched(MockEngine):
        def seed(self, value):
            raise AssertionError("a deterministic engine was seeded")

    result = make_agent(Watched(), tmp_path, seed=8).generate(SCRIPT, seed=8)

    assert result.seed is None


# --- command line ----------------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in list(os.environ):
        if name.startswith("TTS_"):
            monkeypatch.delenv(name)
    logger = logging.getLogger("tts_agent")
    saved = (logger.level, logger.propagate, list(logger.handlers))
    yield
    logger.setLevel(saved[0])
    logger.propagate = saved[1]
    logger.handlers[:] = saved[2]


def test_cli_seed_is_used_and_printed(cli_env, capsys):
    engine = RandomEngine()

    assert main(["--text", FIRST, "--seed", "77", "--quiet"], engine=engine) == EXIT_OK

    assert engine.seeds == [take_seed(77, FIRST, 1)]
    assert "Seed:      77 (pass --seed 77 to repeat these takes)" in capsys.readouterr().out


def test_cli_prints_the_random_seed(cli_env, capsys):
    assert main(["--text", FIRST, "--quiet"], engine=RandomEngine()) == EXIT_OK

    assert "Seed:      " in capsys.readouterr().out


def test_cli_seed_with_a_deterministic_voice_says_it_does_nothing(cli_env, capsys):
    assert main(["--text", FIRST, "--seed", "5", "--quiet"], engine=MockEngine()) == EXIT_OK

    captured = capsys.readouterr()
    assert "Seed:" not in captured.out
    assert "--seed has no effect" in captured.err


def test_cli_without_seed_and_deterministic_voice_prints_nothing_about_seeds(cli_env, capsys):
    assert main(["--text", FIRST, "--quiet"], engine=MockEngine()) == EXIT_OK

    captured = capsys.readouterr()
    assert "eed" not in captured.out + captured.err


@pytest.mark.parametrize("value", ["-3", "abc"])
def test_cli_rejects_a_bad_seed(cli_env, capsys, value):
    assert main(["--text", FIRST, "--seed", value], engine=RandomEngine()) == EXIT_USAGE_ERROR
