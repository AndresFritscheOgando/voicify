from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tts_agent import cli
from tts_agent.cli import (
    EXIT_GENERATION_ERROR,
    EXIT_OK,
    EXIT_PARTIAL,
    EXIT_USAGE_ERROR,
    main,
)
from tts_agent.engines.mock_engine import FailingMockEngine, MockEngine

TEXT = "Welcome to our channel. Today we test the command line."


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Run each test in an empty folder with no TTS_* variables, and restore logging after."""
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


@pytest.fixture
def engine() -> MockEngine:
    return MockEngine()


def run(argv: list[str], engine=None) -> int:
    return main(argv, engine=engine if engine is not None else MockEngine())


def single_error_line(stderr: str) -> str:
    lines = [line for line in stderr.splitlines() if line.startswith("error:")]
    assert len(lines) == 1, stderr
    assert "Traceback" not in stderr
    return lines[0]


# --- success paths ---------------------------------------------------------------------------


def test_text_to_wav(engine, capsys):
    code = run(["--text", TEXT, "--output", "out/hello.wav"], engine)

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert Path("out/hello.wav").exists()
    assert "Saved:     out" in out and "hello.wav" in out
    assert "Voice:     af_heart" in out
    assert "Duration:" in out
    assert "Generated:" in out and "(1 chunk)" in out
    assert engine.calls[0][0] == TEXT


def test_file_input(engine, capsys):
    Path("script.txt").write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")

    assert run(["--file", "script.txt", "--output", "story.wav"], engine) == EXIT_OK
    assert [c[0] for c in engine.calls] == ["First paragraph.", "Second paragraph."]
    assert sf.info("story.wav").samplerate == 24_000


def test_default_output_path():
    assert run(["--text", TEXT]) == EXIT_OK
    assert Path("outputs/narration.wav").exists()


def test_output_dir_from_environment(monkeypatch):
    monkeypatch.setenv("TTS_OUTPUT_DIR", "renders")

    assert run(["--text", TEXT]) == EXIT_OK
    assert Path("renders/narration.wav").exists()


def test_output_without_extension_gets_format(capsys):
    assert run(["--text", TEXT, "--output", "clip", "--format", "wav"]) == EXIT_OK
    assert Path("clip.wav").exists()


def test_voice_and_speed_are_passed_through(engine):
    assert run(["--text", TEXT, "--voice", "BF_Emma"], engine) == EXIT_OK
    assert run(["--text", TEXT, "--voice", "am_michael", "--speed", "1.5"], engine) == EXIT_OK

    assert engine.calls[0][1:] == ("bf_emma", 1.0)
    assert engine.calls[1][1:] == ("am_michael", 1.5)


def test_existing_output_gets_suffix_unless_overwrite(capsys):
    Path("a.wav").write_bytes(b"keep")

    assert run(["--text", TEXT, "--output", "a.wav"]) == EXIT_OK
    assert Path("a.wav").read_bytes() == b"keep"
    assert Path("a_1.wav").exists()
    assert "a_1.wav" in capsys.readouterr().out

    assert run(["--text", TEXT, "--output", "a.wav", "--overwrite"]) == EXIT_OK
    assert Path("a.wav").read_bytes() != b"keep"
    assert not Path("a_2.wav").exists()


def test_list_voices(engine, capsys):
    code = run(["--list-voices"], engine)

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "* af_heart" in out
    assert "  am_michael" in out
    assert "British" in out
    assert "TTS_DEFAULT_VOICE=af_heart" in out
    assert engine.calls == []


def test_list_voices_marks_configured_default(monkeypatch, capsys):
    monkeypatch.setenv("TTS_DEFAULT_VOICE", "bf_emma")

    run(["--list-voices"])

    assert "* bf_emma" in capsys.readouterr().out


def test_no_arguments_prints_help(capsys):
    assert main([]) == EXIT_OK
    assert "usage: voicify" in capsys.readouterr().out


def test_help_includes_examples(capsys):
    assert main(["--help"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "examples:" in out
    assert "--list-voices" in out
    assert "exit codes:" in out


def test_version(capsys):
    assert main(["--version"]) == EXIT_OK
    assert "voicify 0.1.0" in capsys.readouterr().out


def test_device_flag_overrides_settings(monkeypatch):
    seen = {}

    class SpyAgent(cli.TTSAgent):
        def __init__(self, engine=None, settings=None):
            seen["device"] = settings.device
            super().__init__(engine=MockEngine(), settings=settings)

    monkeypatch.setattr(cli, "TTSAgent", SpyAgent)
    monkeypatch.setenv("TTS_DEVICE", "cuda")

    assert main(["--list-voices", "--device", "cpu"]) == EXIT_OK
    assert seen["device"] == "cpu"


def write_voice_sample(path: Path, seconds: float = 6.0) -> Path:
    rate = 24_000
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    sf.write(path, 0.3 * np.sin(2 * np.pi * 180 * t), rate)
    return path


def test_voice_sample_flags_override_settings(monkeypatch):
    seen = {}

    class SpyAgent(cli.TTSAgent):
        def __init__(self, engine=None, settings=None):
            seen["settings"] = settings
            super().__init__(engine=MockEngine(), settings=settings)

    monkeypatch.setattr(cli, "TTSAgent", SpyAgent)
    monkeypatch.setenv("TTS_EXAGGERATION", "1.5")

    argv = ["--list-voices", "--voice-sample", "me.wav", "--exaggeration", "0.7"]
    assert main([*argv, "--cfg-weight", "0.2"]) == EXIT_OK
    settings = seen["settings"]
    assert (settings.voice_sample, settings.exaggeration, settings.cfg_weight) == (
        Path("me.wav"),
        0.7,
        0.2,
    )


def test_list_voices_with_a_voice_sample_shows_the_clone(capsys):
    write_voice_sample(Path("narrator.wav"))

    assert main(["--list-voices", "--voice-sample", "narrator.wav"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "* clone" in out
    assert "Cloned from narrator.wav" in out
    assert "TTS_DEFAULT_VOICE" not in out


def test_generation_with_a_voice_sample_reports_the_clone(capsys):
    from tts_agent.engines.chatterbox_engine import ChatterboxEngine

    class FakeClone(ChatterboxEngine):
        def prepare(self, voice):
            pass

        def synthesize(self, text, voice, speed=1.0):
            return MockEngine().synthesize(text, "af_heart", speed)

    engine = FakeClone(write_voice_sample(Path("narrator.wav")))

    assert run(["--text", TEXT, "--voice-sample", "narrator.wav"], engine) == EXIT_OK
    assert "Voice:     cloned from narrator.wav" in capsys.readouterr().out


# --- logging ---------------------------------------------------------------------------------


def test_default_logging_shows_progress(capsys):
    run(["--text", TEXT])

    assert "generate_start" in capsys.readouterr().err


def test_quiet_hides_info_logs(capsys):
    assert run(["--text", TEXT, "--quiet"]) == EXIT_OK

    captured = capsys.readouterr()
    assert "generate_start" not in captured.err
    assert "Saved:" in captured.out


def test_verbose_and_quiet_are_exclusive(capsys):
    assert run(["--text", TEXT, "--verbose", "--quiet"]) == EXIT_USAGE_ERROR


# --- usage and validation errors (exit 2) ----------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--voice", "af_heart"], "one of --text, --file or --list-voices is required"),
        (["--text", "hi", "--file", "a.txt"], "not allowed with argument"),
        (["--text", ""], "Text is empty"),
        (["--text", "   \n  "], "Text is empty"),
        (["--text", "hi", "--voice", "af_hart"], "Did you mean: af_heart"),
        (["--text", "hi", "--voice", "xx_nobody"], "--list-voices"),
        (["--text", "hi", "--speed", "3"], "speed must be between 0.5 and 2.0"),
        (["--text", "hi", "--speed", "0.1"], "speed must be between 0.5 and 2.0"),
        (["--text", "hi", "--speed", "fast"], "invalid float value"),
        (["--text", "hi", "--format", "ogg"], "invalid choice"),
        (["--text", "hi", "--output", "a.ogg"], "Unsupported output extension '.ogg'"),
        (["--text", "hi", "--output", "a.ogg", "--format", "wav"], "Unsupported output extension"),
        (["--text", "hi", "--device", "tpu"], "invalid choice"),
        (["--file", "missing.txt"], "Input file not found"),
        (["--bogus"], "unrecognized arguments"),
        (["--text", "hi", "--exaggeration", "0.7"], "add --voice-sample"),
        (["--text", "hi", "--cfg-weight", "0.3"], "add --voice-sample"),
        (["--text", "hi", "--voice-sample", "x.wav", "--voice", "am_adam"], "cannot be combined"),
        (["--text", "hi", "--voice-sample", "x.wav", "--exaggeration", "5"], "between 0.25"),
    ],
)
def test_usage_errors(argv, message, capsys):
    assert run(argv) == EXIT_USAGE_ERROR
    err = capsys.readouterr().err
    assert message in err
    assert "Traceback" not in err


def test_file_with_wrong_extension(capsys):
    Path("script.md").write_text("Hello.", encoding="utf-8")

    assert run(["--file", "script.md"]) == EXIT_USAGE_ERROR
    assert "must be a .txt file" in single_error_line(capsys.readouterr().err)


def test_empty_file(capsys):
    Path("empty.txt").write_text(" \n\n ", encoding="utf-8")

    assert run(["--file", "empty.txt"]) == EXIT_USAGE_ERROR
    assert "is empty" in single_error_line(capsys.readouterr().err)


def test_file_that_is_not_utf8(capsys):
    Path("latin1.txt").write_bytes("Café crème".encode("latin-1"))

    assert run(["--file", "latin1.txt"]) == EXIT_USAGE_ERROR
    assert "not valid UTF-8" in single_error_line(capsys.readouterr().err)


def test_file_that_is_a_directory(capsys):
    Path("folder.txt").mkdir()

    assert run(["--file", "folder.txt"]) == EXIT_USAGE_ERROR
    assert "not a file" in single_error_line(capsys.readouterr().err)


def test_invalid_environment_value(monkeypatch, capsys):
    monkeypatch.setenv("TTS_DEVICE", "tpu")

    assert run(["--text", "hi"]) == EXIT_USAGE_ERROR
    assert "TTS_DEVICE" in single_error_line(capsys.readouterr().err)


def test_validation_happens_before_synthesis(engine):
    run(["--text", "hi", "--voice", "nope"], engine)
    run(["--text", "hi", "--speed", "9"], engine)

    assert engine.calls == []
    assert not Path("outputs").exists()


# --- generation errors (exit 1) and partial success (exit 3) ---------------------------------


def test_generation_error_is_one_line(capsys):
    code = run(["--text", TEXT], FailingMockEngine(fail_on=("",)))

    err = capsys.readouterr().err
    assert code == EXIT_GENERATION_ERROR
    assert "All 1 chunk(s) failed" in single_error_line(err)
    assert not Path("outputs").exists()


def test_verbose_shows_traceback(capsys):
    code = run(["--text", TEXT, "--verbose"], FailingMockEngine(fail_on=("",)))

    err = capsys.readouterr().err
    assert code == EXIT_GENERATION_ERROR
    assert "Traceback" in err
    assert "error: All 1 chunk(s) failed" in err


def test_unexpected_exception_is_reported(capsys):
    class BrokenEngine(MockEngine):
        def list_voices(self):
            raise RuntimeError("catalog exploded")

    code = run(["--text", TEXT], BrokenEngine())

    line = single_error_line(capsys.readouterr().err)
    assert code == EXIT_GENERATION_ERROR
    assert "unexpected RuntimeError: catalog exploded" in line
    assert "--verbose" in line


def test_partial_failure_exits_3(monkeypatch, capsys):
    monkeypatch.setenv("TTS_MAX_CHUNK_CHARS", "50")
    text = "The first chunk is fine and says hello. The second chunk will fail to render."
    code = run(["--text", text], FailingMockEngine(fail_on=("second",)))

    captured = capsys.readouterr()
    assert code == EXIT_PARTIAL
    assert Path("outputs/narration.wav").exists()
    assert "Saved:" in captured.out
    assert "warning: 1 of 2 chunks failed and were skipped: 2" in captured.err


def test_keyboard_interrupt(capsys):
    class InterruptedEngine(MockEngine):
        def synthesize(self, text, voice, speed=1.0):
            raise KeyboardInterrupt

    assert run(["--text", TEXT], InterruptedEngine()) == cli.EXIT_INTERRUPTED
    assert "interrupted" in capsys.readouterr().err


# --- MP3 and subtitles -----------------------------------------------------------------------


def test_srt_is_written_by_default(capsys):
    assert run(["--text", TEXT]) == EXIT_OK

    out = capsys.readouterr().out
    assert Path("outputs/narration.srt").exists()
    assert "Subtitles:" in out and "narration.srt" in out


def test_no_srt(capsys):
    assert run(["--text", TEXT, "--no-srt"]) == EXIT_OK

    assert not Path("outputs/narration.srt").exists()
    assert "Subtitles:" not in capsys.readouterr().out


def test_subtitles_disabled_by_environment(monkeypatch):
    monkeypatch.setenv("TTS_SUBTITLES", "false")

    assert run(["--text", TEXT]) == EXIT_OK
    assert not Path("outputs/narration.srt").exists()


def test_format_mp3(capsys):
    assert run(["--text", TEXT, "--format", "mp3"]) == EXIT_OK

    assert sf.info("outputs/narration.mp3").format == "MP3"
    assert Path("outputs/narration.srt").exists()
    assert not Path("outputs/narration.wav").exists()
    assert "narration.mp3" in capsys.readouterr().out


def test_mp3_inferred_from_output_extension():
    assert run(["--text", TEXT, "--output", "ep/one.mp3"]) == EXIT_OK

    assert sf.info("ep/one.mp3").format == "MP3"
    assert Path("ep/one.srt").exists()


def test_keep_wav(capsys):
    assert run(["--text", TEXT, "--format", "mp3", "--keep-wav"]) == EXIT_OK

    out = capsys.readouterr().out
    assert Path("outputs/narration.mp3").exists()
    assert Path("outputs/narration.wav").exists()
    assert "WAV copy:" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["--text", TEXT, "--keep-wav"],
        ["--text", TEXT, "--format", "wav", "--keep-wav"],
        ["--text", TEXT, "--output", "a.wav", "--format", "mp3"],
    ],
)
def test_mp3_option_errors(argv, capsys):
    assert run(argv) == EXIT_USAGE_ERROR
    single_error_line(capsys.readouterr().err)
    assert not Path("outputs").exists()


def test_missing_lameenc_is_a_generation_error(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "lameenc", None)

    assert run(["--text", TEXT, "--format", "mp3"]) == EXIT_GENERATION_ERROR
    assert "lameenc" in single_error_line(capsys.readouterr().err)


# --- script controls -------------------------------------------------------------------------


def test_pronunciations_flag(engine):
    Path("words.txt").write_text("SQL = sequel\n", encoding="utf-8")

    assert run(["--text", "I use SQL.", "--pronunciations", "words.txt"], engine) == EXIT_OK
    assert engine.calls[0][0] == "I use sequel."


def test_pronunciations_from_environment(monkeypatch, engine):
    Path("words.txt").write_text("SQL = sequel\n", encoding="utf-8")
    monkeypatch.setenv("TTS_PRONUNCIATIONS", "words.txt")

    assert run(["--text", "I use SQL."], engine) == EXIT_OK
    assert engine.calls[0][0] == "I use sequel."


@pytest.mark.parametrize(
    ("content", "message"),
    [(None, "Pronunciation file not found"), ("SQL sequel\n", "words.txt:1:")],
)
def test_bad_pronunciation_file_is_a_usage_error(content, message, engine, capsys):
    if content is not None:
        Path("words.txt").write_text(content, encoding="utf-8")

    assert run(["--text", "Hi.", "--pronunciations", "words.txt"], engine) == EXIT_USAGE_ERROR
    assert message in single_error_line(capsys.readouterr().err)
    assert engine.calls == []


def test_pause_markers_from_the_command_line(engine, capsys):
    assert run(["--text", "One. [pause 2s] Two."], engine) == EXIT_OK
    assert [c[0] for c in engine.calls] == ["One.", "Two."]


def test_help_explains_pause_markers(capsys):
    main(["--help"])
    out = capsys.readouterr().out
    assert "[pause 1.5s]" in out and "--pronunciations" in out


# --- --no-clone ------------------------------------------------------------------------------


def test_no_clone_turns_off_a_voice_sample_from_the_environment(monkeypatch, capsys):
    write_voice_sample(Path("narrator.wav"))
    monkeypatch.setenv("TTS_VOICE_SAMPLE", "narrator.wav")
    engine = MockEngine()

    code = run(["--text", TEXT, "--voice", "am_michael", "--no-clone"], engine)

    assert code == EXIT_OK
    assert engine.calls[0][1] == "am_michael"
    assert "Voice:     am_michael" in capsys.readouterr().out


def test_no_clone_selects_kokoro_as_the_default_engine(monkeypatch):
    seen = {}

    class SpyAgent(cli.TTSAgent):
        def __init__(self, engine=None, settings=None):
            seen["settings"] = settings
            super().__init__(engine=MockEngine(), settings=settings)

    monkeypatch.setattr(cli, "TTSAgent", SpyAgent)
    monkeypatch.setenv("TTS_VOICE_SAMPLE", "narrator.wav")

    assert main(["--list-voices", "--no-clone"]) == EXIT_OK
    assert seen["settings"].voice_sample is None


def test_voice_with_an_environment_sample_suggests_no_clone(monkeypatch, capsys):
    write_voice_sample(Path("narrator.wav"))
    monkeypatch.setenv("TTS_VOICE_SAMPLE", "narrator.wav")

    assert run(["--text", TEXT, "--voice", "am_michael"]) == EXIT_USAGE_ERROR
    assert "--no-clone" in single_error_line(capsys.readouterr().err)


def test_no_clone_and_voice_sample_are_exclusive(capsys):
    assert main(["--text", TEXT, "--no-clone", "--voice-sample", "x.wav"]) == EXIT_USAGE_ERROR
    assert "not allowed with" in capsys.readouterr().err


def test_no_clone_rejects_clone_only_options(capsys):
    assert run(["--text", TEXT, "--no-clone", "--exaggeration", "0.7"]) == EXIT_USAGE_ERROR
    assert "only apply to voice cloning" in single_error_line(capsys.readouterr().err)


# --- internal errors -------------------------------------------------------------------------


def test_chunking_bug_is_reported_as_one_clear_line(monkeypatch, capsys):
    from tts_agent import agent as agent_module
    from tts_agent.errors import ChunkingError

    def broken_chunking(text, max_chars):
        raise ChunkingError("Splitting the text lost some of it. This is a bug in Voicify.")

    monkeypatch.setattr(agent_module, "chunk_text", broken_chunking)

    assert run(["--text", TEXT]) == EXIT_GENERATION_ERROR
    line = single_error_line(capsys.readouterr().err)
    assert "bug in Voicify" in line
    assert "unexpected" not in line
