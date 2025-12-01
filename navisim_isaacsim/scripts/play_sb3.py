# scripts/play_sb3.py

"""Evaluate trained Stable-Baselines3 policy on NaviSim navigation environment."""

import argparse
import os
import torch

from isaaclab.app import AppLauncher

# ========================================
# 1. Parse arguments
# ========================================
parser = argparse.ArgumentParser(description="Evaluate SB3 trained policy")
parser.add_argument("--task", type=str, default="Isaac-NavisimNavigation-Jetbot-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.zip)")
parser.add_argument("--video", action="store_true", help="Record evaluation video")
parser.add_argument("--video_length", type=int, default=500, help="Length of video in steps")

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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_rl.utils import parse_env_cfg


def main():
    """Evaluate trained SB3 policy."""

    # ========================================
    # 4. Parse environment config
    # ========================================
    env_cfg = parse_env_cfg(
        task_name=args.task,
        device=args.device if hasattr(args, 'device') else "cuda:0",
        num_envs=args.num_envs,
    )

    # ========================================
    # 5. Create environment
    # ========================================
    env = gym.make(
        args.task,
        cfg=env_cfg,
        render_mode="human" if not args.headless else None
    )

    # Wrap for SB3
    env = Sb3VecEnvWrapper(env)

    # Load normalization stats if they exist
    vecnormalize_path = os.path.join(os.path.dirname(args.checkpoint), "vecnormalize.pkl")
    if os.path.exists(vecnormalize_path):
        print(f"[INFO] Loading VecNormalize from: {vecnormalize_path}")
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False  # Disable training mode
        env.norm_reward = False  # Don't normalize rewards during evaluation

    # ========================================
    # 6. Load trained policy
    # ========================================
    print(f"\n[INFO] Loading checkpoint: {args.checkpoint}")
    model = PPO.load(args.checkpoint, env=env)

    print(f"[INFO] Policy loaded successfully!")
    print(f"  - Task: {args.task}")
    print(f"  - Num envs: {env.num_envs}")
    print(f"  - Device: model.device\n")

    # ========================================
    # 7. Run evaluation
    # ========================================
    obs = env.reset()
    step_count = 0
    episode_rewards = torch.zeros(args.num_envs)
    episode_lengths = torch.zeros(args.num_envs)

    print(f"[INFO] Starting evaluation...")

    while simulation_app.is_running():
        # Get action from policy (deterministic for evaluation)
        actions, _ = model.predict(obs, deterministic=True)

        # Step environment
        obs, rewards, dones, infos = env.step(actions)

        # Track metrics
        episode_rewards += torch.from_numpy(rewards)
        episode_lengths += 1
        step_count += 1

        # Print episode completion
        for idx, done in enumerate(dones):
            if done:
                print(f"[Episode Complete] Env {idx}: "
                      f"Reward = {episode_rewards[idx].item():.3f}, "
                      f"Length = {int(episode_lengths[idx].item())}")
                episode_rewards[idx] = 0
                episode_lengths[idx] = 0

        # Stop after video length if recording
        if args.video and step_count >= args.video_length:
            break

    # ========================================
    # 8. Cleanup
    # ========================================
    print(f"\n[INFO] Evaluation complete! Total steps: {step_count}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
