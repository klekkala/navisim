"""
Gymnasium RL Environment Module

Provides Gymnasium-compatible RL environment for navigation training.

Usage:
    from navisim_isaac.gymnasium import NaviSimIsaacEnv

    env = NaviSimIsaacEnv(
        sequence_graph_path='assets/sequence_graph.gpickle',
        config={'max_episode_steps': 500, 'headless': True}
    )

    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
"""

from .env import NaviSimIsaacEnv

try:
    from .agent import NavigationAgent
    from .rl import RLTrainer
    __all__ = ["NaviSimIsaacEnv", "NavigationAgent", "RLTrainer"]
except ImportError:
    # agent and rl modules may not be fully implemented yet
    __all__ = ["NaviSimIsaacEnv"]
