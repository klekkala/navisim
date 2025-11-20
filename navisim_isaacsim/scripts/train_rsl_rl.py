# scripts/train_rsl_rl.py

"""Train NaviSim navigation environment using RSL-RL (PPO)."""

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

# ========================================
# 1. Parse arguments (before importing Isaac modules)
# ========================================
parser = argparse.ArgumentParser(description="Train NaviSim navigation with RSL-RL PPO")
parser.add_argument("--task", type=str, default="Isaac-NavisimNavigation-Jetbot-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum training iterations")
parser.add_argument("--run_name", type=str, default="", help="Name of the run")
parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"], help="Logger type")
parser.add_argument("--video", action="store_true", help="Record videos during training")
parser.add_argument("--video_interval", type=int, default=1000, help="Video recording interval (steps)")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume")

# AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ========================================
# 2. Launch Isaac Sim
# ========================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ========================================
# 3. Import modules AFTER AppLauncher
# ========================================
import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner
from isaaclab_rl.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.dict import print_dict


def main():
    """Train with RSL-RL PPO algorithm."""

    # ========================================
    # 4. Parse environment and agent configs
    # ========================================
    # Get environment registry
    env_cfg = parse_env_cfg(
        task_name=args.task,
        device=args.device if hasattr(args, 'device') else "cuda:0",
        num_envs=args.num_envs,
    )

    # Get agent config from registry
    agent_cfg = gym.spec(args.task).kwargs.get("rsl_rl_cfg_entry_point")
    if isinstance(agent_cfg, str):
        # String entry point - import and instantiate
        module_path, class_name = agent_cfg.rsplit(":", 1)
        module = __import__(module_path, fromlist=[class_name])
        agent_cfg = getattr(module, class_name)()
    else:
        # Already a class - instantiate
        agent_cfg = agent_cfg()

    # Override agent config with CLI arguments
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.run_name:
        agent_cfg.run_name = args.run_name

    # ========================================
    # 5. Create environment
    # ========================================
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)

    # Print configuration
    print("\n" + "="*80)
    print("[INFO] Environment Configuration:")
    print("="*80)
    print_dict(env_cfg.to_dict(), nesting=2)

    print("\n" + "="*80)
    print("[INFO] Agent Configuration:")
    print("="*80)
    print_dict(agent_cfg.__dict__, nesting=2)
    print("="*80 + "\n")

    # ========================================
    # 6. Setup logging directory
    # ========================================
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_dir = os.path.join(
        log_root_path,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ("_" + args.run_name if args.run_name else "")
    )
    os.makedirs(log_dir, exist_ok=True)

    # ========================================
    # 7. Create RSL-RL runner
    # ========================================
    runner = RslRlOnPolicyRunner(
        env=env,
        cfg=agent_cfg,
        log_dir=log_dir,
        device=env.device,
    )

    # ========================================
    # 8. Load checkpoint if specified
    # ========================================
    if args.checkpoint:
        checkpoint_path = get_checkpoint_path(log_root_path, args.checkpoint)
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        runner.load(checkpoint_path)

    # ========================================
    # 9. Train the agent
    # ========================================
    print(f"\n[INFO] Starting training...")
    print(f"  - Task: {args.task}")
    print(f"  - Num envs: {env.num_envs}")
    print(f"  - Max iterations: {agent_cfg.max_iterations}")
    print(f"  - Device: {env.device}")
    print(f"  - Log directory: {log_dir}\n")

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # ========================================
    # 10. Cleanup
    # ========================================
    print("\n[INFO] Training complete!")
    print(f"  - Final model saved to: {log_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
