"""Base interface for RL agent frameworks in NaviSim."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Any
import torch


class BaseRLAgent(ABC):
    """Abstract base class for RL agent implementations.

    This interface allows different RL frameworks (RSL-RL, Stable-Baselines3, skrl, etc.)
    to be used interchangeably with NaviSim environments.
    """

    @abstractmethod
    def setup(self, env: Any, config: dict) -> "BaseRLAgent":
        """Initialize agent with environment and configuration.

        Args:
            env: The Isaac Lab environment (may be wrapped by this method)
            config: Agent configuration dictionary

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def train(self, num_iterations: int, log_dir: Path) -> None:
        """Train the agent.

        Args:
            num_iterations: Number of training iterations
            log_dir: Directory to save logs and checkpoints
        """
        pass

    @abstractmethod
    def load(self, checkpoint_path: Path) -> None:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        pass

    @abstractmethod
    def save(self, save_path: Path) -> None:
        """Save model checkpoint.

        Args:
            save_path: Path to save checkpoint
        """
        pass

    @abstractmethod
    def get_policy(self, deterministic: bool = True) -> Callable:
        """Get inference policy function.

        Args:
            deterministic: Whether to use deterministic policy

        Returns:
            Callable that takes observations and returns actions
        """
        pass

    @property
    @abstractmethod
    def requires_env_wrapper(self) -> bool:
        """Whether this agent requires environment wrapping.

        Returns:
            True if wrap_env() should be called before setup()
        """
        pass

    @abstractmethod
    def wrap_env(self, env: Any) -> Any:
        """Apply agent-specific environment wrapper.

        Args:
            env: Original environment

        Returns:
            Wrapped environment
        """
        pass

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, checkpoint_path: Path, config: dict) -> "BaseRLAgent":
        """Create agent instance from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            config: Agent configuration

        Returns:
            Initialized agent with loaded weights
        """
        pass

    def get_device(self) -> str:
        """Get the device being used by the agent.

        Returns:
            Device string (e.g., 'cuda:0', 'cpu')
        """
        if hasattr(self, 'env') and hasattr(self.env, 'unwrapped'):
            return str(self.env.unwrapped.device)
        return "cuda:0"  # Default fallback
