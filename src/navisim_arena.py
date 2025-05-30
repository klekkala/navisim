from agents.navisim_agent import NavisimAgent
from envs.navisim_env import NavisimEnv


class NavisimArena:
    """
    Arena class to orchestrate interaction between NavisimEnv and NavisimAgent.

    Responsibilities:
    - Reset the environment
    - Let the agent interact with the environment
    - Render each step (optional)
    - Track total reward across the episode
    """

    def __init__(self, env : NavisimEnv, agent : NavisimAgent):
        """
        Initialize the arena with an environment and an agent.

        Args:
            env: A Gymnasium-compatible environment (e.g., NavisimEnv).
            agent: A decision-making agent with an `act(observation)` method.
        """
        self.env = env
        self.agent = agent

    def run_episode(self, max_steps: int = 100, render: bool = True) -> float:
        """
        Run a single episode in the arena.

        Args:
            max_steps (int): Maximum number of steps in the episode.
            render (bool): Whether to render the environment at each step.

        Returns:
            float: Total accumulated reward for the episode.
        """
        observation, _ = self.env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            action = self.agent.act(observation)
            observation, reward, done, truncated, info = self.env.step(action)
            total_reward += reward

            if render:
                self.env.render()

            if done:
                break

        return total_reward