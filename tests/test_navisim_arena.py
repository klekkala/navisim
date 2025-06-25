import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import MagicMock, patch
from navisim.navisim_arena import NavisimArena 


@pytest.fixture
def mock_env():
    mock_env = MagicMock()
    mock_env.reset.return_value = ("observation", {"info": "reset"})
    mock_env.action_space.sample.return_value = "action"
    mock_env.step.return_value = ("observation", 1.0, False, False, {"info": "step"})
    mock_env.render.return_value = ("render_frame", 0.05)  # 20 FPS
    return mock_env


@pytest.fixture
def mock_logger():
    with patch("navisim.utils.resource_logger.ResourceLogger") as MockLogger:
        logger_instance = MockLogger.return_value
        logger_instance.log = MagicMock()
        yield MockLogger


def test_run_episode_accumulates_reward(mock_env, mock_logger):
    arena = NavisimArena(env=mock_env)
    total_reward = arena.run_episode(max_steps=5, render=True, on_render=lambda *args: None)

    # Expected reward = 1.0 * 5
    assert total_reward == 5.0

    # Ensure environment methods are called correctly
    assert mock_env.reset.called
    assert mock_env.step.call_count == 5
    assert mock_env.render.call_count == 5

    # Check logger was called with expected values
    assert mock_logger.return_value.log.call_count == 5


def test_run_episode_without_render(mock_env, mock_logger):
    arena = NavisimArena(env=mock_env)
    total_reward = arena.run_episode(max_steps=3, render=False)

    assert total_reward == 3.0
    assert mock_env.render.call_count == 0
    assert mock_logger.return_value.log.call_count == 0


def test_run_episode_calls_on_render(mock_env, mock_logger):
    on_render_mock = MagicMock()
    arena = NavisimArena(env=mock_env)
    arena.run_episode(max_steps=2, render=True, on_render=on_render_mock)

    assert on_render_mock.call_count == 2