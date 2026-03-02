#!/usr/bin/env python3
"""Play/evaluate a trained PPO agent in the Navisim warehouse environment.

Usage:
    python scripts/rsl_rl/play.py --checkpoint logs/rsl_rl/warehouse_jetbot/ppo/model_8.pt
    python scripts/rsl_rl/play.py --checkpoint logs/model.pt --save_camera_images --save_every 10
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
NAVISIM_ISAACSIM_DIR = SCRIPT_DIR.parents[1]
if str(NAVISIM_ISAACSIM_DIR) not in sys.path:
    sys.path.insert(0, str(NAVISIM_ISAACSIM_DIR))

from isaaclab.app import AppLauncher

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Play/evaluate trained PPO agent")
    parser.add_argument("--task", type=str, default="Navisim-Warehouse-Jetbot-v0", help="Task name")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to model checkpoint")
    parser.add_argument("--num_steps", type=int, default=50000, help="Number of steps to run")
    parser.add_argument(
        "--agent_cfg",
        type=str,
        default=str(NAVISIM_ISAACSIM_DIR / "navisim_lab" / "configs" / "rsl_rl" / "ppo_warehouse_jetbot.yaml"),
        help="Path to agent config YAML",
    )
    parser.add_argument("--save_camera_images", action="store_true", help="Save Jetbot POV images during playback")
    parser.add_argument("--save_every", type=int, default=10, help="Save camera images every N steps")
    parser.add_argument("--output_dir", type=str, default="outputs/playback_pov", help="Output directory for images")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Playing trained agent for Navisim Warehouse Navigation")

    simulation_app = None
    env = None

    try:
        logger.info(f"Task: {args.task} | Envs: {args.num_envs} | Steps: {args.num_steps}")

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

        # Setup camera image saving
        output_path = None
        save_camera_batch = None
        if args.save_camera_images:
            from navisim_lab.utils.camera_utils import save_camera_batch
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving camera images to: {output_path}")

        # Create environment
        env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
        env = gym.make(args.task, cfg=env_cfg)
        logger.info(f"Environment created: obs={env.observation_space}, act={env.action_space}")

        # Wrap env for RSL-RL
        env = RslRlVecEnvWrapper(env)
        env_device = env.unwrapped.device

        # Load checkpoint and policy
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            logger.info(f"Loading checkpoint: {checkpoint_path}")
            agent_cfg = OmegaConf.load(args.agent_cfg)
            agent_cfg_dict = OmegaConf.to_container(agent_cfg, resolve=True)

            runner = OnPolicyRunner(env, agent_cfg_dict, log_dir="")
            runner.load(str(checkpoint_path))
            policy = runner.get_inference_policy(device=env_device)
            logger.info(f"Policy loaded on device: {env_device}")
        else:
            logger.warning("No checkpoint provided, using zero policy for testing")

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

            # Save camera images if requested
            if save_camera_batch is not None and step % args.save_every == 0:
                try:
                    save_camera_batch(
                        env.unwrapped,
                        output_path,
                        camera_name="jetbot_camera",
                        prefix="playback_pov",
                        step=step,
                    )
                    if step % 100 == 0:
                        logger.info(f"Saved camera images at step {step}")
                except Exception as e:
                    if step == 0:
                        logger.warning(f"Could not save camera images: {e}")

            if dones.any():
                logger.info(f"Episode ended at step {step + 1}")
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
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
