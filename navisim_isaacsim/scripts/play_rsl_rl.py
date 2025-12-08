# scripts/play_rsl_rl.py

"""Evaluate trained RSL-RL policy on NaviSim navigation environment."""

import argparse
import os
import torch

from isaaclab.app import AppLauncher

# ========================================
# 1. Parse arguments
# ========================================
parser = argparse.ArgumentParser(description="Evaluate RSL-RL trained policy")
parser.add_argument("--task", type=str, default="Isaac-NavisimNavigation-Jetbot-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
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
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner
from isaaclab_rl.utils import parse_env_cfg, get_checkpoint_path


def main():
    """Evaluate trained RSL-RL policy."""

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

    # ========================================
    # 6. Load trained policy
    # ========================================
    print(f"\n[INFO] Loading checkpoint: {args.checkpoint}")

    # Get agent config
    agent_cfg_entry = gym.spec(args.task).kwargs.get("rsl_rl_cfg_entry_point")
    if isinstance(agent_cfg_entry, str):
        module_path, class_name = agent_cfg_entry.rsplit(":", 1)
        module = __import__(module_path, fromlist=[class_name])
        agent_cfg = getattr(module, class_name)()
    else:
        agent_cfg = agent_cfg_entry()

    # Create runner and load policy
    runner = RslRlOnPolicyRunner(env=env, cfg=agent_cfg, log_dir=None, device=env.device)
    runner.load(args.checkpoint)

    print(f"[INFO] Policy loaded successfully!")
    print(f"  - Task: {args.task}")
    print(f"  - Num envs: {env.num_envs}")
    print(f"  - Device: {env.device}\n")

    # ========================================
    # 7. Run evaluation
    # ========================================
    obs, _ = env.reset()
    step_count = 0
    episode_rewards = torch.zeros(args.num_envs, device=env.device)
    episode_lengths = torch.zeros(args.num_envs, device=env.device)

    print(f"[INFO] Starting evaluation...")

    while simulation_app.is_running():
        # Get action from policy
        with torch.no_grad():
            actions = runner.get_inference_action(obs["policy"])

        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Track metrics
        episode_rewards += rewards
        episode_lengths += 1
        step_count += 1

        # Print episode completion
        dones = terminated | truncated
        if dones.any():
            done_indices = torch.where(dones)[0]
            for idx in done_indices:
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
