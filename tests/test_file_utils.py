from __future__ import annotations

import pytest

from tts_agent.errors import OutputError
from tts_agent.utils.file_utils import resolve_output_path


def test_new_path_is_returned_unchanged(tmp_path):
    assert resolve_output_path(tmp_path / "a.wav") == tmp_path / "a.wav"


def test_missing_parent_directories_are_created(tmp_path):
    target = tmp_path / "x" / "y" / "a.wav"

    assert resolve_output_path(target) == target
    assert target.parent.is_dir()
    assert not target.exists()


def test_existing_file_gets_numeric_suffix(tmp_path):
    (tmp_path / "narration.wav").touch()
    assert resolve_output_path(tmp_path / "narration.wav") == tmp_path / "narration_1.wav"

    (tmp_path / "narration_1.wav").touch()
    assert resolve_output_path(tmp_path / "narration.wav") == tmp_path / "narration_2.wav"


def test_suffix_skips_gaps_only_when_taken(tmp_path):
    (tmp_path / "narration.wav").touch()
    (tmp_path / "narration_2.wav").touch()

    assert resolve_output_path(tmp_path / "narration.wav") == tmp_path / "narration_1.wav"


def test_overwrite_returns_existing_path(tmp_path):
    (tmp_path / "narration.wav").touch()

    assert resolve_output_path(tmp_path / "narration.wav", overwrite=True) == (
        tmp_path / "narration.wav"
    )


def test_accepts_string_paths(tmp_path):
    assert resolve_output_path(str(tmp_path / "a.wav")) == tmp_path / "a.wav"


def test_directory_path_is_rejected(tmp_path):
    with pytest.raises(OutputError):
        resolve_output_path(tmp_path)


def test_parent_that_is_a_file_is_rejected(tmp_path):
    (tmp_path / "blocker").write_text("not a dir")

    with pytest.raises(OutputError):
        resolve_output_path(tmp_path / "blocker" / "a.wav")
