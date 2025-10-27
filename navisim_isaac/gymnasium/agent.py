"""
Navigation Agent Module

Agent interface for navigation tasks.
"""

from typing import Any, Dict, Optional


class NavigationAgent:
    """
    Navigation agent that receives step commands and generates actions.

    This agent:
    - Processes observations from the environment
    - Generates navigation actions
    - Maintains internal state for navigation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize NavigationAgent.

        Args:
            config: Agent configuration dictionary
        """
        self.config = config or {}
        self.state = None

    def select_action(self, observation: Any) -> Any:
        """
        Select action based on current observation.

        Args:
            observation: Current observation from environment

        Returns:
            Action to take
        """
        raise NotImplementedError("select_action not yet implemented")

    def update(self, observation: Any, action: Any, reward: float, next_observation: Any, done: bool) -> None:
        """
        Update agent based on environment feedback.

        Args:
            observation: Previous observation
            action: Action taken
            reward: Reward received
            next_observation: New observation
            done: Whether episode is done
        """
        raise NotImplementedError("update not yet implemented")

    def save(self, path: str) -> None:
        """
        Save agent to file.

        Args:
            path: Path to save agent
        """
        raise NotImplementedError("save not yet implemented")

    def load(self, path: str) -> None:
        """
        Load agent from file.

        Args:
            path: Path to load agent from
        """
        raise NotImplementedError("load not yet implemented")

    def reset(self) -> None:
        """
        Reset agent state.
        """
        self.state = None
