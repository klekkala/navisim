# scripts/train_sb3.py

"""Train NaviSim navigation environment using Stable-Baselines3 (PPO)."""

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

# ========================================
# 1. Parse arguments (before importing Isaac modules)
# ========================================
parser = argparse.ArgumentParser(description="Train NaviSim navigation with Stable-Baselines3 PPO")
parser.add_argument("--task", type=str, default="Isaac-NavisimNavigation-Jetbot-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--total_timesteps", type=int, default=None, help="Total training timesteps")
parser.add_argument("--run_name", type=str, default="", help="Name of the run")
parser.add_argument("--video", action="store_true", help="Record videos during training")
parser.add_argument("--video_interval", type=int, default=10000, help="Video recording interval (steps)")
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
import yaml
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_rl.utils import parse_env_cfg


def main():
    """Train with Stable-Baselines3 PPO algorithm."""

    # ========================================
    # 4. Parse environment configuration
    # ========================================
    env_cfg = parse_env_cfg(
        task_name=args.task,
        device=args.device if hasattr(args, 'device') else "cuda:0",
        num_envs=args.num_envs,
    )

    # ========================================
    # 5. Load agent configuration from YAML
    # ========================================
    agent_cfg_entry = gym.spec(args.task).kwargs.get("sb3_cfg_entry_point")

    if isinstance(agent_cfg_entry, str):
        # Extract YAML path from entry point
        if ":" in agent_cfg_entry:
            module_path, file_name = agent_cfg_entry.rsplit(":", 1)
            module = __import__(module_path, fromlist=["AGENTS_DIR"])
            agents_dir = getattr(module, "AGENTS_DIR", Path(module.__file__).parent)
            agent_cfg_path = agents_dir / file_name
        else:
            agent_cfg_path = Path(agent_cfg_entry)

        with open(agent_cfg_path, "r") as f:
            agent_cfg = yaml.safe_load(f)
    else:
        raise ValueError("SB3 config must be a YAML file path")

    # Override with CLI arguments
    if args.total_timesteps is not None:
        agent_cfg["n_timesteps"] = args.total_timesteps
    if args.seed is not None:
        agent_cfg["seed"] = args.seed

    # ========================================
    # 6. Create environment
    # ========================================
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)

    # Wrap for Stable-Baselines3
    env = Sb3VecEnvWrapper(env)

    # Apply normalization if specified
    if agent_cfg.get("normalize_input", False):
        env = VecNormalize(
            env,
            training=True,
            norm_obs=True,
            norm_reward=agent_cfg.get("normalize_value", True),
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=agent_cfg.get("gamma", 0.99),
        )

    print("\n" + "="*80)
    print("[INFO] Training Configuration:")
    print("="*80)
    print(f"  Task: {args.task}")
    print(f"  Num envs: {env.num_envs}")
    print(f"  Total timesteps: {agent_cfg['n_timesteps']}")
    print(f"  Seed: {agent_cfg['seed']}")
    print(f"  Device: {agent_cfg.get('device', 'cuda')}")
    print("="*80 + "\n")

    # ========================================
    # 7. Setup logging directory
    # ========================================
    log_root_path = os.path.join("logs", "sb3", args.task.replace("Isaac-", ""))
    log_dir = os.path.join(
        log_root_path,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ("_" + args.run_name if args.run_name else "")
    )
    os.makedirs(log_dir, exist_ok=True)

    # Configure SB3 logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    # ========================================
    # 8. Create or load PPO model
    # ========================================
    if args.checkpoint:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        model = PPO.load(
            args.checkpoint,
            env=env,
            device=agent_cfg.get("device", "cuda"),
        )
        model.set_logger(new_logger)
    else:
        # Create new model
        policy_kwargs = agent_cfg.get("policy", {})

        model = PPO(
            policy=policy_kwargs.get("class_type", "MlpPolicy"),
            env=env,
            learning_rate=agent_cfg.get("learning_rate", 3e-4),
            n_steps=agent_cfg.get("n_steps", 2048),
            batch_size=agent_cfg.get("batch_size", 64),
            n_epochs=agent_cfg.get("n_epochs", 10),
            gamma=agent_cfg.get("gamma", 0.99),
            gae_lambda=agent_cfg.get("gae_lambda", 0.95),
            clip_range=agent_cfg.get("clip_range", 0.2),
            ent_coef=agent_cfg.get("ent_coef", 0.0),
            vf_coef=agent_cfg.get("vf_coef", 0.5),
            max_grad_norm=agent_cfg.get("max_grad_norm", 0.5),
            verbose=agent_cfg.get("verbose", 1),
            tensorboard_log=log_dir,
            device=agent_cfg.get("device", "cuda"),
            seed=agent_cfg["seed"],
        )
        model.set_logger(new_logger)

    # ========================================
    # 9. Setup callbacks
    # ========================================
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10k steps
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # ========================================
    # 10. Train the agent
    # ========================================
    print(f"\n[INFO] Starting training...")
    print(f"  - Log directory: {log_dir}\n")

    model.learn(
        total_timesteps=agent_cfg["n_timesteps"],
        callback=checkpoint_callback,
        log_interval=10,
    )

    # ========================================
    # 11. Save final model
    # ========================================
    model_path = os.path.join(log_dir, "model")
    model.save(model_path)
    print(f"\n[INFO] Training complete!")
    print(f"  - Final model saved to: {model_path}.zip")

    # Save VecNormalize statistics
    if isinstance(env, VecNormalize):
        env.save(os.path.join(log_dir, "vecnormalize.pkl"))

    # ========================================
    # 12. Cleanup
    # ========================================
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
