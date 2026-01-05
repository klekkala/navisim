#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

from isaaclab.app import AppLauncher

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Parse arguments BEFORE launching app
def parse_args():
    parser = argparse.ArgumentParser(description="Play/evaluate trained PPO agent for Navisim warehouse navigation")
    parser.add_argument("--task", type=str, default="Navisim-Warehouse-Jetbot-v0", help="Task name")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to model checkpoint (e.g., model_500.pt)")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to run")
    parser.add_argument(
        "--agent_cfg",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "navisim_lab" / "configs" / "rsl_rl" / "ppo_warehouse_jetbot.yaml"),
        help="Path to agent config YAML",
    )
    # AppLauncher adds its own arguments (--headless, --enable_cameras, etc.)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Playing trained agent for Navisim Warehouse Navigation")

    simulation_app = None
    env = None

    try:
        logger.info(f"Task: {args.task} | Envs: {args.num_envs} | Steps: {args.num_steps}")

        # Enable cameras and launch Isaac Sim
        args.enable_cameras = True
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app

        # Import modules after AppLauncher
        import gymnasium as gym
        import torch
        from omegaconf import OmegaConf
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg
        import navisim_lab.tasks  # noqa: F401 - registers task
        from rsl_rl.runners import OnPolicyRunner

        # Check CUDA availability and warn user
        if not torch.cuda.is_available():
            logger.warning("CUDA not available! Isaac Sim requires CUDA. Ensure container has GPU access (--gpus all)")
            logger.warning("Attempting to override device to 'cpu' but Isaac Sim may not support CPU-only mode")

        # Create environment
        env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)

        # Override device to CPU if CUDA unavailable (may not work with Isaac Sim)
        if not torch.cuda.is_available() and hasattr(env_cfg, 'sim') and hasattr(env_cfg.sim, 'device'):
            logger.warning("Overriding sim.device to 'cpu'")
            env_cfg.sim.device = "cpu"

        env = gym.make(args.task, cfg=env_cfg)
        logger.info(f"Environment created: obs={env.observation_space}, act={env.action_space}")

        # Wrap env for RSL-RL
        env = RslRlVecEnvWrapper(env)

        # Get the actual device from the environment
        env_device = env.unwrapped.device
        logger.info(f"Environment is using device: {env_device}")

        # Load checkpoint and policy
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            logger.info(f"Loading checkpoint: {checkpoint_path}")

            # Load agent config
            agent_cfg = OmegaConf.load(args.agent_cfg)
            agent_cfg_dict = OmegaConf.to_container(agent_cfg, resolve=True)

            # Create runner and load checkpoint
            runner = OnPolicyRunner(env, agent_cfg_dict, log_dir="")
            runner.load(str(checkpoint_path))

            # Use the same device as the environment (not the config device)
            policy = runner.get_inference_policy(device=env_device)
            logger.info(f"Policy loaded successfully on device: {env_device}")
        else:
            logger.warning("No checkpoint provided, using zero policy for testing")
            # Simple zero policy for testing
            def policy(obs):
                return torch.zeros((obs.shape[0], env.num_actions), device=obs.device, dtype=obs.dtype)

        # Run rollout
        logger.info(f"Starting rollout for {args.num_steps} steps...")
        obs, _ = env.reset()
        total_reward = 0.0

        for step in range(args.num_steps):
            with torch.inference_mode():
                actions = policy(obs)

            obs, rewards, dones, info = env.step(actions)
            total_reward += rewards.mean().item()

            if dones.any():
                logger.info(f"Episode ended at step {step+1}")
                break

        logger.info(f"Rollout completed! Total reward: {total_reward:.2f}")

    except KeyboardInterrupt:
        logger.warning("Playback interrupted by user")
    except Exception as e:
        logger.error(f"Playback failed: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        # Cleanup
        if env is not None:
            try:
                env.close()
            except Exception as e:
                logger.error(f"Error closing environment: {e}")

        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception as e:
                logger.error(f"Error closing Isaac Sim: {e}")


if __name__ == "__main__":
    main()
