#!/usr/bin/env python3
"""Unified inference/play script for NaviSim with pluggable RL agents.

This script supports loading and evaluating trained models from multiple
RL frameworks (RSL-RL, Stable-Baselines3, skrl, etc.).

Example usage:
    # Play with RSL-RL checkpoint (auto-detect agent type)
    python scripts/play.py --task Navisim-Warehouse-Jetbot-v0 --checkpoint logs/model.pt --config navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml

    # Specify agent explicitly
    python scripts/play.py --task Navisim-Warehouse-Jetbot-v0 --agent rsl_rl --checkpoint logs/model.pt --config my_config.yaml --num_envs 1

    # Run headless for benchmarking
    python scripts/play.py --task Navisim-Warehouse-Jetbot-v0 --checkpoint logs/model.pt --config my_config.yaml --headless --num_steps 10000
"""

import argparse
import logging
import sys
from pathlib import Path

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
        description="Play/evaluate trained RL agent for NaviSim"
    )

    # Task and checkpoint
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Gym task ID (e.g., Navisim-Warehouse-Jetbot-v0)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., model_500.pt)",
    )

    # Agent selection (optional, will auto-detect if not provided)
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Agent framework (rsl_rl, sb3, skrl, etc.). Auto-detected if not specified.",
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

    # Inference parameters
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of parallel environments"
    )
    parser.add_argument(
        "--num_steps", type=int, default=5000, help="Number of steps to run"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (default: True for most agents)",
    )

    # AppLauncher arguments (--headless, --enable_cameras, etc.)
    AppLauncher.add_app_launcher_args(parser)

    return parser.parse_args()


def detect_agent_from_checkpoint(checkpoint_path: Path) -> str:
    """Auto-detect agent type from checkpoint metadata.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Agent type string (e.g., 'rsl_rl', 'sb3')

    Raises:
        ValueError: If agent type cannot be determined
    """
    import torch

    logger.info(f"Auto-detecting agent type from checkpoint: {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Check for explicit metadata
        if isinstance(ckpt, dict):
            if "metadata" in ckpt and "agent_type" in ckpt["metadata"]:
                agent_type = ckpt["metadata"]["agent_type"]
                logger.info(f"Found agent type in metadata: {agent_type}")
                return agent_type

            # Heuristic detection based on checkpoint structure
            if "ac_parameters_state_dict" in ckpt or "model_state_dict" in ckpt:
                logger.info("Detected RSL-RL checkpoint structure")
                return "rsl_rl"
            elif "policy" in ckpt:
                logger.info("Detected Stable-Baselines3 checkpoint structure")
                return "sb3"

        raise ValueError("Cannot determine agent type from checkpoint structure")

    except Exception as e:
        raise ValueError(
            f"Failed to auto-detect agent type: {e}. "
            "Please specify --agent explicitly."
        )


def main():
    """Main inference loop."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("NaviSim Unified Inference Script")
    logger.info("=" * 80)
    logger.info(f"Task:       {args.task}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Num Envs:   {args.num_envs}")
    logger.info(f"Num Steps:  {args.num_steps}")
    logger.info("=" * 80)

    simulation_app = None
    env = None

    try:
        # Verify checkpoint exists
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Auto-detect agent type if not specified
        if args.agent is None:
            args.agent = detect_agent_from_checkpoint(checkpoint_path)
            logger.info(f"Auto-detected agent: {args.agent}")

        # Launch Isaac Sim
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app

        # Import after AppLauncher
        import gymnasium as gym
        import torch
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

        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available! Isaac Sim requires CUDA.")
            logger.warning(
                "Ensure container has GPU access (docker run --gpus all ...)"
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
        config_dict = OmegaConf.to_container(config, resolve=True)

        # Create agent
        logger.info(f"Creating agent: {args.agent} ({args.algorithm})")
        agent = create_agent(args.agent, algorithm=args.algorithm)

        # Wrap environment if needed
        if agent.requires_env_wrapper:
            logger.info("Wrapping environment with agent-specific wrapper")
            env = agent.wrap_env(env)

        # Setup agent
        logger.info("Setting up agent with environment")
        agent.setup(env, config_dict)

        # Load checkpoint
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        agent.load(checkpoint_path)

        # Get policy
        logger.info("Getting inference policy")
        policy = agent.get_policy(deterministic=True)

        # Get device
        device = agent.get_device()
        logger.info(f"Running on device: {device}")

        logger.info("=" * 80)
        logger.info(f"Starting rollout for {args.num_steps} steps...")
        logger.info("=" * 80)

        # Run rollout
        obs, _ = env.reset()
        total_reward = 0.0
        episode_count = 0

        for step in range(args.num_steps):
            with torch.inference_mode():
                actions = policy(obs)

            obs, rewards, dones, info = env.step(actions)
            total_reward += rewards.mean().item()

            # Log episode completions
            if dones.any():
                episode_count += dones.sum().item()
                logger.info(
                    f"Step {step+1}/{args.num_steps}: "
                    f"{episode_count} episodes completed, "
                    f"avg reward: {total_reward/(step+1):.3f}"
                )

            # Progress logging every 100 steps
            if (step + 1) % 100 == 0:
                logger.info(
                    f"Progress: {step+1}/{args.num_steps} steps, "
                    f"avg reward: {total_reward/(step+1):.3f}"
                )

        logger.info("=" * 80)
        logger.info("Rollout completed successfully!")
        logger.info(f"Total steps:    {args.num_steps}")
        logger.info(f"Episodes:       {episode_count}")
        logger.info(f"Total reward:   {total_reward:.2f}")
        logger.info(f"Average reward: {total_reward/args.num_steps:.4f}")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("Playback interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Playback failed: {type(e).__name__}: {str(e)}")
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
