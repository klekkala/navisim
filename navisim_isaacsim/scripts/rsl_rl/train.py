#!/usr/bin/env python3
import argparse
import os
import logging
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

# Parse arguments BEFORE launching app
def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent with RSL-RL for Navisim warehouse navigation")
    parser.add_argument("--task", type=str, default="Navisim-Warehouse-Jetbot-v0", help="Task name")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--agent_cfg",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "navisim_lab" / "configs" / "rsl_rl" / "ppo_warehouse_jetbot.yaml"),
        help="Path to agent config YAML",
    )
    parser.add_argument("--log_dir", type=str, default="check_pts/rsl_rl", help="Log directory")
    # AppLauncher adds its own arguments (--headless, --enable_cameras, etc.)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Starting RSL-RL Training for Navisim Warehouse Navigation")

    simulation_app = None
    env = None

    try:
        logger.info(f"Task: {args.task} | Envs: {args.num_envs} | Seed: {args.seed}")

        # Enable cameras and launch Isaac Sim
        args.enable_cameras = True
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app

        # Import modules after AppLauncher
        import gymnasium as gym
        from omegaconf import OmegaConf
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg
        import navisim_lab.tasks  # noqa: F401 - registers task

        # Create environment
        env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
        env = gym.make(args.task, cfg=env_cfg)
        logger.info(f"Environment created: obs={env.observation_space}, act={env.action_space}")

        # Load agent config
        agent_cfg = OmegaConf.load(args.agent_cfg)
        agent_cfg.seed = args.seed
        agent_cfg_dict = OmegaConf.to_container(agent_cfg, resolve=True)

        # Wrap env and create runner
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg_dict.get("clip_actions", None))

        from rsl_rl.runners import OnPolicyRunner
        log_dir = Path(args.log_dir) / agent_cfg_dict.get("experiment_name", "exp") / agent_cfg_dict.get("run_name", "run")
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving logs to: {log_dir}")

        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=str(log_dir))

        # Train
        max_iters = int(agent_cfg_dict["max_iterations"])
        logger.info(f"Starting training for {max_iters} iterations...")
        runner.learn(num_learning_iterations=max_iters, init_at_random_ep_len=True)

        logger.info(f"Training completed! Model saved to: {log_dir}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {type(e).__name__}: {str(e)}")
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
