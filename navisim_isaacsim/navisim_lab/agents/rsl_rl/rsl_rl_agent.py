"""RSL-RL agent implementation for NaviSim."""

import logging
from pathlib import Path
from typing import Callable, Optional, Any
import torch

from navisim_lab.agents.base_agent import BaseRLAgent
from navisim_lab.agents import register_agent

logger = logging.getLogger(__name__)


@register_agent("rsl_rl")
class RslRlAgent(BaseRLAgent):
    """Wrapper for RSL-RL (Robot Supervised Learning - RL) framework.

    RSL-RL is NVIDIA's optimized RL library designed for robotics,
    supporting on-policy algorithms like PPO.

    Args:
        algorithm: Algorithm name (default: "ppo")
    """

    SUPPORTED_ALGORITHMS = ["ppo"]  # Can be extended with SAC, etc.

    def __init__(self, algorithm: str = "ppo"):
        """Initialize RSL-RL agent.

        Args:
            algorithm: Algorithm name (currently only 'ppo' supported)
        """
        if algorithm.lower() not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Algorithm '{algorithm}' not supported by RSL-RL. "
                f"Supported: {self.SUPPORTED_ALGORITHMS}"
            )

        self.algorithm = algorithm.lower()
        self.runner: Optional[Any] = None
        self.env: Optional[Any] = None
        self.config: Optional[dict] = None

    def setup(self, env: Any, config: dict) -> "RslRlAgent":
        """Initialize agent with environment and configuration.

        Args:
            env: Isaac Lab environment (will be wrapped)
            config: Configuration dictionary containing RSL-RL parameters

        Returns:
            Self for method chaining
        """
        # Import here to avoid issues if RSL-RL not installed
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        # Wrap environment for RSL-RL compatibility
        self.env = RslRlVecEnvWrapper(env)
        self.config = config

        logger.info(f"RSL-RL agent setup complete (algorithm: {self.algorithm})")
        return self

    def train(self, num_iterations: int, log_dir: Path) -> None:
        """Train the agent using RSL-RL.

        Args:
            num_iterations: Number of training iterations
            log_dir: Directory to save logs and checkpoints
        """
        from rsl_rl.runners import OnPolicyRunner

        if self.env is None or self.config is None:
            raise RuntimeError("Agent not setup. Call setup() before train()")

        # Create log directory
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting RSL-RL training for {num_iterations} iterations")
        logger.info(f"Logs will be saved to: {log_dir}")

        # Create runner
        self.runner = OnPolicyRunner(
            env=self.env, train_cfg=self.config, log_dir=str(log_dir)
        )

        # Train
        self.runner.learn(
            num_learning_iterations=num_iterations, init_at_random_ep_len=True
        )

        logger.info("Training completed!")

    def load(self, checkpoint_path: Path) -> None:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to RSL-RL checkpoint file
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if self.runner is None:
            # Create runner for inference
            from rsl_rl.runners import OnPolicyRunner

            if self.env is None or self.config is None:
                raise RuntimeError(
                    "Agent not setup. Call setup() before load() or use from_checkpoint()"
                )

            self.runner = OnPolicyRunner(
                env=self.env, train_cfg=self.config, log_dir=""
            )

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        self.runner.load(str(checkpoint_path))
        logger.info("Checkpoint loaded successfully")

    def save(self, save_path: Path) -> None:
        """Save model checkpoint.

        Args:
            save_path: Path to save checkpoint
        """
        if self.runner is None:
            raise RuntimeError("No trained model to save. Train first or load a checkpoint.")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.runner.save(str(save_path))
        logger.info(f"Checkpoint saved to: {save_path}")

    def get_policy(self, deterministic: bool = True) -> Callable:
        """Get inference policy function.

        Args:
            deterministic: Whether to use deterministic policy (ignored for RSL-RL)

        Returns:
            Callable that takes observations and returns actions
        """
        if self.runner is None:
            raise RuntimeError("No trained model. Train first or load a checkpoint.")

        device = self.get_device()
        policy = self.runner.get_inference_policy(device=device)

        logger.info(f"Policy ready for inference on device: {device}")
        return policy

    @property
    def requires_env_wrapper(self) -> bool:
        """RSL-RL requires environment wrapping.

        Returns:
            True
        """
        return True

    def wrap_env(self, env: Any) -> Any:
        """Apply RSL-RL environment wrapper.

        Args:
            env: Original Isaac Lab environment

        Returns:
            RslRlVecEnvWrapper instance
        """
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        return RslRlVecEnvWrapper(env)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: Path, config: dict, algorithm: str = "ppo"
    ) -> "RslRlAgent":
        """Create agent from checkpoint without environment.

        Note: This creates an agent instance but setup() must still be called
        with an environment before inference.

        Args:
            checkpoint_path: Path to checkpoint file
            config: Agent configuration
            algorithm: Algorithm name

        Returns:
            Initialized agent (call setup() before use)
        """
        agent = cls(algorithm=algorithm)
        agent.config = config

        logger.info(f"Agent created from checkpoint: {checkpoint_path}")
        logger.info("Note: Call setup(env, config) before using the agent")

        return agent

    def get_device(self) -> str:
        """Get the device being used by the environment.

        Returns:
            Device string (e.g., 'cuda:0', 'cpu')
        """
        if self.env is not None and hasattr(self.env, "unwrapped"):
            return str(self.env.unwrapped.device)
        return "cuda:0"  # Default fallback
