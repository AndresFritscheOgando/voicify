from __future__ import annotations

import pytest

from tts_agent.agent import TTSAgent
from tts_agent.config import Settings
from tts_agent.engines.mock_engine import MockEngine


@pytest.fixture
def mock_engine() -> MockEngine:
    return MockEngine()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(output_dir=tmp_path / "outputs")


@pytest.fixture
def agent(mock_engine, settings) -> TTSAgent:
    return TTSAgent(engine=mock_engine, settings=settings)
