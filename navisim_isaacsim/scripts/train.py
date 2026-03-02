#!/usr/bin/env python3
"""Unified training script for NaviSim with pluggable RL agents.

This script supports multiple RL frameworks (RSL-RL, Stable-Baselines3, skrl, etc.)
through a unified agent registry system.

Example usage:
    # Train with RSL-RL PPO
    python scripts/train.py --task Navisim-Warehouse-Jetbot-v0 --agent rsl_rl --algorithm ppo --config navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml

    # Train with custom settings
    python scripts/train.py --task Navisim-Warehouse-Jetbot-v0 --agent rsl_rl --config my_config.yaml --num_envs 128 --max_iterations 2000
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Fix for newer GPUs (Blackwell/GB10) not recognized by PyTorch JIT
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
os.environ.setdefault("PYTORCH_JIT", "0")

# Add navisim_isaacsim to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
NAVISIM_ISAACSIM_DIR = SCRIPT_DIR.parent
if str(NAVISIM_ISAACSIM_DIR) not in sys.path:
    sys.path.insert(0, str(NAVISIM_ISAACSIM_DIR))

from isaaclab.app import AppLauncher

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL agent for NaviSim navigation tasks"
    )

    # Task and agent selection
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Gym task ID (e.g., Navisim-Warehouse-Jetbot-v0)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="rsl_rl",
        help="Agent framework (rsl_rl, sb3, skrl, etc.)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        help="Algorithm name (ppo, sac, td3, etc.)",
    )

    # Configuration
    parser.add_argument(
        "--config", type=str, required=True, help="Path to agent configuration YAML"
    )

    # Training parameters
    parser.add_argument(
        "--num_envs", type=int, default=64, help="Number of parallel environments"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Maximum training iterations (overrides config)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Logging
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Base directory for logs and checkpoints",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name (default: task name)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name (default: agent_algorithm)",
    )

    # AppLauncher arguments (--headless, --enable_cameras, etc.)
    AppLauncher.add_app_launcher_args(parser)

    return parser.parse_args()


def main():
    """Main training loop."""
    args = parse_args()

    # Set defaults for experiment organization
    if args.experiment_name is None:
        args.experiment_name = args.task
    if args.run_name is None:
        args.run_name = f"{args.agent}_{args.algorithm}"

    logger.info("=" * 80)
    logger.info("NaviSim Unified Training Script")
    logger.info("=" * 80)
    logger.info(f"Task:       {args.task}")
    logger.info(f"Agent:      {args.agent}")
    logger.info(f"Algorithm:  {args.algorithm}")
    logger.info(f"Config:     {args.config}")
    logger.info(f"Num Envs:   {args.num_envs}")
    logger.info(f"Seed:       {args.seed}")
    logger.info("=" * 80)

    simulation_app = None
    env = None

    try:
        # Launch Isaac Sim
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app

        # Import after AppLauncher (required for Isaac Sim initialization)
        import gymnasium as gym
        from omegaconf import OmegaConf
        from isaaclab_tasks.utils import parse_env_cfg
        import navisim_lab.tasks  # noqa: F401 - registers gym tasks
        from navisim_lab.agents import create_agent, list_agents

        # Verify agent is available
        available_agents = list_agents()
        if args.agent not in available_agents:
            raise ValueError(
                f"Agent '{args.agent}' not found. Available: {available_agents}"
            )

        logger.info(f"Creating environment: {args.task}")

        # Create environment
        env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
        env = gym.make(args.task, cfg=env_cfg)

        logger.info(f"Environment created successfully")
        logger.info(f"  Observation space: {env.observation_space}")
        logger.info(f"  Action space:      {env.action_space}")

        # Load agent configuration
        logger.info(f"Loading agent config from: {args.config}")
        config = OmegaConf.load(args.config)

        # Override config parameters from command line
        config.seed = args.seed
        if args.max_iterations is not None:
            config.max_iterations = args.max_iterations

        # Convert to dict
        config_dict = OmegaConf.to_container(config, resolve=True)

        # Create agent
        logger.info(f"Creating agent: {args.agent} ({args.algorithm})")
        agent = create_agent(args.agent, algorithm=args.algorithm)

        # Wrap environment if needed
        if agent.requires_env_wrapper:
            logger.info("Wrapping environment with agent-specific wrapper")
            env = agent.wrap_env(env)

        # Setup agent with environment and config
        logger.info("Setting up agent with environment")
        agent.setup(env, config_dict)

        # Determine log directory
        log_path = Path(args.log_dir) / args.experiment_name / args.run_name
        log_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Logs and checkpoints will be saved to: {log_path}")

        # Get max iterations from config if not overridden
        max_iterations = (
            args.max_iterations
            if args.max_iterations is not None
            else config_dict.get("max_iterations", 1000)
        )

        logger.info("=" * 80)
        logger.info(f"Starting training for {max_iterations} iterations...")
        logger.info("=" * 80)

        # Train
        agent.train(num_iterations=max_iterations, log_dir=log_path)

        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {log_path}")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Training failed: {type(e).__name__}: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        raise
    finally:
        # Cleanup
        if env is not None:
            try:
                logger.info("Closing environment...")
                env.close()
            except Exception as e:
                logger.error(f"Error closing environment: {e}")

        if simulation_app is not None:
            try:
                logger.info("Closing Isaac Sim...")
                simulation_app.close()
            except Exception as e:
                logger.error(f"Error closing Isaac Sim: {e}")


if __name__ == "__main__":
    main()
