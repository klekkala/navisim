from pygame import Surface
from agents.navisim_agent import NavisimAgent
from envs.navisim_env import NavisimEnv
from typing import Callable, Optional, Any

import time

class NavisimArena:
    """
    Arena class to orchestrate interaction between NavisimEnv and NavisimAgent.

    Responsibilities:
    - Reset the environment
    - Let the agent interact with the environment
    - Render each step (optional)
    - Track total reward across the episode
    """

    def __init__(self, env : NavisimEnv):
        """
        Initialize the arena with an environment and an agent.

        Args:
            env: A Gymnasium-compatible environment (e.g., NavisimEnv).
            agent: A decision-making agent with an `act(observation)` method.
        """
        self.env = env

    def run_episode(self, max_steps: int = 100, render: bool = True, on_render: Optional[Callable[[Any], None]] = None) -> float:
        """
        Run a single episode in the arena.

        Args:
            max_steps (int): Maximum number of steps in the episode.
            render (bool): Whether to render the environment at each step.

        Returns:
            float: Total accumulated reward for the episode.
        """
        observation, info = self.env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = self.env.action_space.sample()
            observation, reward, done, truncated, info = self.env.step(action)
            print(info)
            
            total_reward += reward

            if render:
                start_time = time.time()

                rendering = self.env.render()
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else float('inf')
                on_render(rendering, step, fps)

        return total_reward