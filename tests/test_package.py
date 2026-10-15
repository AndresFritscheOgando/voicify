from tts_agent import __version__
from tts_agent.cli import main


def test_version_is_set():
    assert __version__


def test_cli_without_args_prints_help(capsys):
    assert main([]) == 0
    assert "voicify" in capsys.readouterr().out
