from __future__ import annotations

import pytest

from tts_agent.errors import EmptyTextError, InputFileError, OutputError
from tts_agent.utils.file_utils import read_text_file, resolve_output_path

# --- read_text_file --------------------------------------------------------------------------


def test_reads_utf8_text(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("Café — “quoted”", encoding="utf-8")

    assert read_text_file(path) == "Café — “quoted”"


def test_strips_byte_order_mark(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes("﻿Hello.".encode())

    assert read_text_file(str(path)) == "Hello."


def test_extension_check_is_case_insensitive(tmp_path):
    path = tmp_path / "A.TXT"
    path.write_text("Hello.", encoding="utf-8")

    assert read_text_file(path) == "Hello."


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("missing.txt", None, "not found"),
        ("script.md", b"Hello.", "must be a .txt file"),
        ("script", b"Hello.", "must be a .txt file"),
        ("latin1.txt", "Café".encode("latin-1"), "not valid UTF-8"),
    ],
)
def test_bad_input_files(tmp_path, name, content, message):
    path = tmp_path / name
    if content is not None:
        path.write_bytes(content)

    with pytest.raises(InputFileError, match=message):
        read_text_file(path)


def test_directory_is_not_a_file(tmp_path):
    (tmp_path / "dir.txt").mkdir()

    with pytest.raises(InputFileError, match="not a file"):
        read_text_file(tmp_path / "dir.txt")


@pytest.mark.parametrize("content", ["", "   \n\n\t"])
def test_empty_file_raises_empty_text_error(tmp_path, content):
    path = tmp_path / "empty.txt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(EmptyTextError, match="empty"):
        read_text_file(path)


# --- resolve_output_path ---------------------------------------------------------------------


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
