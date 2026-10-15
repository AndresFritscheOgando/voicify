from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tts_agent import (
    EmptyTextError,
    InputFileError,
    InvalidOptionError,
    ModelLoadError,
    Settings,
    SynthesisError,
    TTSAgent,
    UnsupportedVoiceError,
)
from tts_agent.engines.mock_engine import FailingMockEngine, MockEngine

EXAMPLE = "Welcome to our channel. Today we are going to explore local text-to-speech."


def test_generate_writes_wav(agent, tmp_path):
    result = agent.generate(EXAMPLE, output_path=tmp_path / "hello.wav")

    assert result.output_path == tmp_path / "hello.wav"
    audio, rate = sf.read(result.output_path, dtype="float32")
    assert rate == 24_000
    assert audio.ndim == 1
    assert np.abs(audio).max() > 0.1
    assert result.duration_seconds == pytest.approx(len(audio) / rate)
    assert result.sample_rate == 24_000
    assert result.chunk_count == 1
    assert result.failed_chunks == []
    assert result.format == "wav"


def test_generate_uses_defaults(agent, mock_engine, settings):
    result = agent.generate(EXAMPLE)

    assert result.output_path == settings.output_dir / "narration.wav"
    assert result.output_path.exists()
    assert result.voice == "af_heart"
    assert mock_engine.calls == [(EXAMPLE, "af_heart", 1.0)]


def test_generate_passes_voice_and_speed(agent, mock_engine, tmp_path):
    result = agent.generate(EXAMPLE, voice="am_michael", speed=1.5, output_path=tmp_path / "a.wav")

    assert result.voice == "am_michael"
    assert mock_engine.calls == [(EXAMPLE, "am_michael", 1.5)]


@pytest.mark.parametrize("text", ["", "   ", "\n\t \n"])
def test_empty_text_raises(agent, mock_engine, settings, text):
    with pytest.raises(EmptyTextError):
        agent.generate(text)
    assert mock_engine.calls == []
    assert not settings.output_dir.exists()


def test_missing_output_directories_are_created(agent, tmp_path):
    target = tmp_path / "deep" / "nested" / "dir" / "out.wav"

    result = agent.generate(EXAMPLE, output_path=target)

    assert result.output_path == target
    assert target.exists()


def test_existing_file_is_not_overwritten(agent, tmp_path):
    target = tmp_path / "narration.wav"
    target.write_bytes(b"keep me")

    first = agent.generate(EXAMPLE, output_path=target)
    second = agent.generate(EXAMPLE, output_path=target)

    assert target.read_bytes() == b"keep me"
    assert first.output_path == tmp_path / "narration_1.wav"
    assert second.output_path == tmp_path / "narration_2.wav"


def test_overwrite_replaces_existing_file(agent, tmp_path):
    target = tmp_path / "narration.wav"
    target.write_bytes(b"old")

    result = agent.generate(EXAMPLE, output_path=target, overwrite=True)

    assert result.output_path == target
    assert target.read_bytes() != b"old"


def test_output_path_without_extension_gets_wav(agent, tmp_path):
    result = agent.generate(EXAMPLE, output_path=tmp_path / "clip")

    assert result.output_path == tmp_path / "clip.wav"


def test_unsupported_output_extension(agent, tmp_path):
    with pytest.raises(InvalidOptionError):
        agent.generate(EXAMPLE, output_path=tmp_path / "clip.ogg")


def test_output_name_with_dots_keeps_them(agent, tmp_path):
    result = agent.generate(EXAMPLE, output_path=tmp_path / "episode.v2.wav")

    assert result.output_path == tmp_path / "episode.v2.wav"


@pytest.mark.parametrize("fmt", ["wav", "WAV", ".wav"])
def test_format_is_normalized(agent, tmp_path, fmt):
    result = agent.generate(EXAMPLE, output_path=tmp_path / "a", format=fmt)

    assert result.output_path == tmp_path / "a.wav"
    assert result.format == "wav"


def test_format_sets_default_file_name(agent, settings):
    result = agent.generate(EXAMPLE, format="wav")

    assert result.output_path == settings.output_dir / "narration.wav"


def test_format_must_be_supported(agent):
    with pytest.raises(InvalidOptionError, match="Unsupported format 'flac'"):
        agent.generate(EXAMPLE, format="flac")


def test_format_must_match_output_extension(agent, tmp_path):
    with pytest.raises(InvalidOptionError, match="does not match format 'mp3'"):
        agent.generate(EXAMPLE, output_path=tmp_path / "a.wav", format="mp3")
    with pytest.raises(InvalidOptionError, match="does not match format 'wav'"):
        agent.generate(EXAMPLE, output_path=tmp_path / "a.mp3", format="wav")


def test_generate_from_file(agent, mock_engine, tmp_path):
    script = tmp_path / "script.txt"
    script.write_text("﻿First paragraph.\n\nSecond paragraph.", encoding="utf-8")

    result = agent.generate(file=script, output_path=tmp_path / "a.wav")

    assert [c[0] for c in mock_engine.calls] == ["First paragraph.", "Second paragraph."]
    assert result.chunk_count == 2


def test_text_and_file_are_exclusive(agent, tmp_path):
    with pytest.raises(InvalidOptionError, match="not both"):
        agent.generate("Hi.", file=tmp_path / "a.txt")


def test_missing_text_and_file(agent):
    with pytest.raises(EmptyTextError):
        agent.generate()


def test_bad_input_file_raises(agent, tmp_path):
    with pytest.raises(InputFileError):
        agent.generate(file=tmp_path / "missing.txt")


def test_spec_api_example_runs_unchanged(mock_engine, tmp_path, monkeypatch):
    """The Python API example from docs/phases.md, with the engine injected."""
    monkeypatch.chdir(tmp_path)
    agent = TTSAgent(engine=mock_engine, settings=Settings())
    result = agent.generate(
        text="Hello, welcome to my channel.",
        voice="default",
        speed=1.0,
        output_path="outputs/hello.wav",
        format="wav",
        overwrite=False,
    )
    _fields = (  # every attribute the spec example reads must exist
        result.output_path,
        result.duration_seconds,
        result.sample_rate,
        result.voice,
        result.generation_seconds,
        result.chunk_count,
        result.failed_chunks,
        result.subtitle_path,
    )

    assert result.output_path == Path("outputs/hello.wav")
    assert (tmp_path / "outputs" / "hello.wav").exists()
    assert result.voice == "af_heart"
    assert result.subtitle_path == Path("outputs/hello.srt")


def test_engine_returning_no_audio_raises(settings, tmp_path):
    class EmptyEngine(MockEngine):
        def synthesize(self, text, voice, speed=1.0):
            return np.zeros(0, dtype=np.float32)

    agent = TTSAgent(engine=EmptyEngine(), settings=settings)
    with pytest.raises(SynthesisError):
        agent.generate(EXAMPLE, output_path=tmp_path / "x.wav")
    assert not (tmp_path / "x.wav").exists()


def test_list_voices_comes_from_engine(agent, mock_engine):
    assert agent.list_voices() == mock_engine.list_voices()


@pytest.mark.parametrize("voice", ["default", None, "DEFAULT", " default "])
def test_default_alias_uses_configured_voice(mock_engine, tmp_path, voice):
    agent = TTSAgent(engine=mock_engine, settings=Settings(default_voice="am_michael"))

    result = agent.generate(EXAMPLE, voice=voice, output_path=tmp_path / "a.wav")

    assert result.voice == "am_michael"
    assert mock_engine.calls[-1][1] == "am_michael"


def test_voice_ids_are_case_insensitive(agent, tmp_path):
    assert agent.generate(EXAMPLE, voice="BF_Emma", output_path=tmp_path / "a.wav").voice == (
        "bf_emma"
    )


def test_unsupported_voice_suggests_close_matches(agent, mock_engine, settings):
    with pytest.raises(UnsupportedVoiceError) as excinfo:
        agent.generate(EXAMPLE, voice="af_hart")

    message = str(excinfo.value)
    assert "'af_hart'" in message
    assert "Did you mean: af_heart" in message
    assert "list_voices()" in message
    assert mock_engine.calls == []
    assert not settings.output_dir.exists()


def test_unsupported_voice_without_close_match(agent):
    with pytest.raises(UnsupportedVoiceError) as excinfo:
        agent.generate(EXAMPLE, voice="zz_nobody")

    assert "Did you mean" not in str(excinfo.value)


def test_bad_default_voice_names_the_setting(mock_engine):
    agent = TTSAgent(engine=mock_engine, settings=Settings(default_voice="af_nope"))

    with pytest.raises(UnsupportedVoiceError, match="TTS_DEFAULT_VOICE"):
        agent.generate(EXAMPLE)


@pytest.mark.parametrize("speed", [0.5, 1, 1.25, 2.0])
def test_valid_speeds_are_passed_to_engine(agent, mock_engine, tmp_path, speed):
    agent.generate(EXAMPLE, speed=speed, output_path=tmp_path / "a.wav")

    assert mock_engine.calls[-1][2] == speed


def test_default_speed_comes_from_settings(mock_engine, tmp_path):
    agent = TTSAgent(engine=mock_engine, settings=Settings(default_speed=1.5))

    agent.generate(EXAMPLE, output_path=tmp_path / "a.wav")

    assert mock_engine.calls[-1][2] == 1.5


@pytest.mark.parametrize("speed", [0.49, 2.01, 0, -1, float("nan"), float("inf"), "fast", True])
def test_invalid_speed_is_rejected(agent, mock_engine, speed):
    with pytest.raises(InvalidOptionError, match="speed"):
        agent.generate(EXAMPLE, speed=speed)
    assert mock_engine.calls == []


def test_faster_speed_gives_shorter_audio(agent, tmp_path):
    slow = agent.generate(EXAMPLE, speed=0.5, output_path=tmp_path / "slow.wav")
    fast = agent.generate(EXAMPLE, speed=2.0, output_path=tmp_path / "fast.wav")

    assert fast.duration_seconds < slow.duration_seconds


# --- long-form: chunking, pauses, failures ---------------------------------------------------

RATE = 24_000
SENTENCE_MS = 250
PARAGRAPH_MS = 700

THREE_CHUNKS = (
    "The first chunk has exactly one sentence here.\n\n"
    "The second chunk opens paragraph number two. "
    "The third chunk closes paragraph number two."
)
SECOND = "The second chunk opens paragraph number two."


def chunked_agent(engine, tmp_path, **overrides) -> TTSAgent:
    options = {
        "output_dir": tmp_path / "outputs",
        "max_chunk_chars": 50,
        "sentence_pause_ms": SENTENCE_MS,
        "paragraph_pause_ms": PARAGRAPH_MS,
    }
    options.update(overrides)
    return TTSAgent(engine=engine, settings=Settings(**options))


def test_chunks_are_synthesized_in_order(tmp_path):
    engine = MockEngine()
    result = chunked_agent(engine, tmp_path).generate(THREE_CHUNKS)

    assert [c[0] for c in engine.calls] == [
        "The first chunk has exactly one sentence here.",
        SECOND,
        "The third chunk closes paragraph number two.",
    ]
    assert {c[1:] for c in engine.calls} == {("af_heart", 1.0)}
    assert result.chunk_count == 3
    assert result.failed_chunks == []


def test_pauses_and_timeline(tmp_path):
    engine = MockEngine()
    result = chunked_agent(engine, tmp_path).generate(THREE_CHUNKS)

    lengths = [len(MockEngine().synthesize(c[0], "af_heart")) for c in engine.calls]
    paragraph_gap = PARAGRAPH_MS * RATE // 1000
    sentence_gap = SENTENCE_MS * RATE // 1000
    audio, _ = sf.read(result.output_path, dtype="float32")
    assert len(audio) == sum(lengths) + paragraph_gap + sentence_gap

    second_start = lengths[0] + paragraph_gap
    third_start = second_start + lengths[1] + sentence_gap
    starts = [0, second_start, third_start]
    assert [t.index for t in result.timeline] == [0, 1, 2]
    for timing, start, length in zip(result.timeline, starts, lengths, strict=True):
        assert timing.start == pytest.approx(start / RATE)
        assert timing.end == pytest.approx((start + length) / RATE)
    assert result.timeline[-1].end == pytest.approx(result.duration_seconds)

    # the gaps really are silent, and the speech around them is not
    assert np.abs(audio[lengths[0] : second_start]).max() == 0
    assert np.abs(audio[third_start - sentence_gap : third_start]).max() == 0
    assert np.abs(audio[: lengths[0]]).max() > 0.1


def test_pause_lengths_are_configurable(tmp_path):
    agent = chunked_agent(MockEngine(), tmp_path, sentence_pause_ms=0, paragraph_pause_ms=1000)
    result = agent.generate(THREE_CHUNKS)

    assert result.timeline[1].start - result.timeline[0].end == pytest.approx(1.0)
    assert result.timeline[2].start == pytest.approx(result.timeline[1].end)


def test_failed_chunk_is_retried_once_then_succeeds(tmp_path):
    engine = FailingMockEngine(fail_on=("second",), failures=1)
    result = chunked_agent(engine, tmp_path).generate(THREE_CHUNKS)

    assert result.failed_chunks == []
    assert len(result.timeline) == 3
    assert engine.attempts == {SECOND: 2}


def test_retry_count_is_configurable(tmp_path):
    engine = FailingMockEngine(fail_on=("second",), failures=2)
    result = chunked_agent(engine, tmp_path, chunk_retries=2).generate(THREE_CHUNKS)
    assert result.failed_chunks == []

    engine = FailingMockEngine(fail_on=("second",))
    result = chunked_agent(engine, tmp_path, chunk_retries=0).generate(THREE_CHUNKS)
    assert result.failed_chunks == [1]
    assert engine.attempts == {SECOND: 1}


def test_failing_chunk_is_skipped_and_reported(tmp_path, caplog):
    engine = FailingMockEngine(fail_on=("third",))
    with caplog.at_level("WARNING", logger="tts_agent"):
        result = chunked_agent(engine, tmp_path).generate(THREE_CHUNKS)

    assert result.failed_chunks == [2]
    assert result.chunk_count == 3
    assert [t.index for t in result.timeline] == [0, 1]
    assert result.output_path.exists()
    assert result.duration_seconds == pytest.approx(result.timeline[-1].end)
    assert "chunk_failed chunk=3/3 attempt=2/2" in caplog.text


def test_skipped_chunk_keeps_the_paragraph_pause(tmp_path):
    text = (
        "The first paragraph has only one chunk.\n\n"
        "The second paragraph opens with this one. Its other chunk is right here now."
    )

    engine = FailingMockEngine(fail_on=("first paragraph",))
    result = chunked_agent(engine, tmp_path).generate(text)
    assert result.chunk_count == 3
    assert result.failed_chunks == [0]
    assert result.timeline[0].start == 0  # no leading pause for the first audio
    assert result.timeline[1].start - result.timeline[0].end == pytest.approx(SENTENCE_MS / 1000)

    engine = FailingMockEngine(fail_on=("second paragraph",))
    result = chunked_agent(engine, tmp_path).generate(text)
    assert result.failed_chunks == [1]
    # chunk 1 failed between chunk 0 and chunk 2, and a paragraph break came before it
    assert result.timeline[1].start - result.timeline[0].end == pytest.approx(PARAGRAPH_MS / 1000)


def test_abort_policy_stops_and_writes_nothing(tmp_path):
    engine = FailingMockEngine(fail_on=("second",))
    agent = chunked_agent(engine, tmp_path, on_chunk_error="abort")

    with pytest.raises(SynthesisError, match="Chunk 2/3 failed"):
        agent.generate(THREE_CHUNKS)
    assert not (tmp_path / "outputs").exists()
    assert len(engine.calls) == 3  # chunk 1, then chunk 2 twice; chunk 3 never runs


def test_abort_can_be_chosen_per_call(tmp_path):
    agent = chunked_agent(FailingMockEngine(fail_on=("second",)), tmp_path)

    with pytest.raises(SynthesisError):
        agent.generate(THREE_CHUNKS, on_chunk_error="abort")


def test_invalid_chunk_error_policy(tmp_path):
    with pytest.raises(InvalidOptionError, match="on_chunk_error"):
        chunked_agent(MockEngine(), tmp_path).generate(THREE_CHUNKS, on_chunk_error="ignore")


def test_all_chunks_failing_raises_and_writes_nothing(tmp_path):
    engine = FailingMockEngine(fail_on=("",))

    with pytest.raises(SynthesisError, match="All 3 chunk"):
        chunked_agent(engine, tmp_path).generate(THREE_CHUNKS)
    assert not (tmp_path / "outputs").exists()


def test_empty_chunk_audio_counts_as_failure(tmp_path):
    class HalfEmptyEngine(MockEngine):
        def synthesize(self, text, voice, speed=1.0):
            audio = super().synthesize(text, voice, speed)
            return audio[:0] if "second" in text else audio

    result = chunked_agent(HalfEmptyEngine(), tmp_path).generate(THREE_CHUNKS)

    assert result.failed_chunks == [1]


def test_model_load_error_is_not_retried_per_chunk(tmp_path):
    class BrokenEngine(MockEngine):
        def synthesize(self, text, voice, speed=1.0):
            self.calls.append((text, voice, speed))
            raise ModelLoadError("no model")

    engine = BrokenEngine()
    with pytest.raises(ModelLoadError):
        chunked_agent(engine, tmp_path).generate(THREE_CHUNKS)
    assert len(engine.calls) == 1


def test_progress_is_logged(tmp_path, caplog):
    with caplog.at_level("INFO", logger="tts_agent"):
        chunked_agent(MockEngine(), tmp_path).generate(THREE_CHUNKS)

    assert "chunk 1/3 (33%)" in caplog.text
    assert "chunk 3/3 (100%)" in caplog.text


def test_long_form_500_sentences(tmp_path):
    sentences = [f"This is sentence number {i}, and it says something." for i in range(500)]
    paragraphs = [" ".join(sentences[i : i + 10]) for i in range(0, 500, 10)]
    engine = MockEngine(seconds_per_char=0.001)
    agent = chunked_agent(engine, tmp_path, max_chunk_chars=400)

    result = agent.generate("\n\n".join(paragraphs))

    assert " ".join(call[0] for call in engine.calls) == " ".join(sentences)
    assert result.chunk_count == len(engine.calls) == len(result.timeline)
    assert result.chunk_count >= 50  # at least one chunk per paragraph
    assert result.failed_chunks == []
    assert all(len(call[0]) <= 400 for call in engine.calls)
    assert all(a.end < b.start for a, b in zip(result.timeline, result.timeline[1:], strict=False))


def test_engine_is_prepared_before_synthesis_and_load_time_is_separate(settings, tmp_path):
    events = []

    class SlowLoadingEngine(MockEngine):
        def prepare(self, voice):
            events.append(("prepare", voice))
            import time

            time.sleep(0.05)

        def synthesize(self, text, voice, speed=1.0):
            events.append(("synthesize", voice))
            return super().synthesize(text, voice, speed)

    result = TTSAgent(engine=SlowLoadingEngine(), settings=settings).generate(
        EXAMPLE, output_path=tmp_path / "a.wav"
    )

    assert events[0] == ("prepare", "af_heart")
    assert events[1][0] == "synthesize"
    assert result.load_seconds >= 0.05
    assert result.generation_seconds < result.load_seconds


def test_model_load_error_from_prepare_stops_before_synthesis(settings, tmp_path):
    class UnloadableEngine(MockEngine):
        def prepare(self, voice):
            raise ModelLoadError("no CUDA")

    engine = UnloadableEngine()
    with pytest.raises(ModelLoadError):
        TTSAgent(engine=engine, settings=settings).generate(EXAMPLE, output_path=tmp_path / "a.wav")
    assert engine.calls == []
    assert not (tmp_path / "a.wav").exists()


def test_default_engine_is_kokoro_on_the_configured_device(settings):
    from dataclasses import replace

    from tts_agent.engines.kokoro_engine import KokoroEngine

    agent = TTSAgent(settings=replace(settings, device="cpu"))

    assert isinstance(agent.engine, KokoroEngine)
    assert agent.engine._requested_device == "cpu"
    assert agent.engine.device is None  # nothing loads until the first generate()
