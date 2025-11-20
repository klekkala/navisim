# scripts/run_with_policy.py

"""Run NaviSim environment with different policies."""

import argparse
import torch
from isaaclab.app import AppLauncher

# ========================================
# 1. Parse arguments
# ========================================
parser = argparse.ArgumentParser(description="Run NaviSim with custom policies")
parser.add_argument("--task", type=str, default="Isaac-NavisimNavigation-Jetbot-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--policy", type=str, default="forward",
                   help="Policy name (forward, circle, random, waypoint, obstacle_avoidance, rsl_rl, sb3, hybrid)")
parser.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint for learned policies")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum simulation steps")
parser.add_argument("--video", action="store_true", help="Record video")
parser.add_argument("--video_interval", type=int, default=500, help="Video recording interval")

# Policy-specific arguments
parser.add_argument("--action_scale", type=float, default=0.8, help="Action scale for scripted policies")
parser.add_argument("--pattern", type=str, default="forward_turn",
                   choices=["forward_turn", "spiral"], help="Motion pattern for waypoint policy")

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
from configs.navigation_env_cfg import NavisimNavigationEnvCfg
from tasks.navigation_env import NavisimNavigationEnv
from policies import get_policy, list_policies
from policies.scripted_policies import WaypointPolicyConfig
from policies.learned_policies import LearnedPolicyConfig


def main():
    """Run environment with specified policy."""

    # ========================================
    # 4. Create environment
    # ========================================
    env_cfg = NavisimNavigationEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = NavisimNavigationEnv(
        cfg=env_cfg,
        render_mode="human" if not args.headless else None
    )

    print(f"\n{'='*80}")
    print(f"[NaviSim] Environment Created")
    print(f"{'='*80}")
    print(f"  Task: {args.task}")
    print(f"  Num envs: {args.num_envs}")
    print(f"  Observation space: {env.single_observation_space}")
    print(f"  Action space: {env.single_action_space}")
    print(f"{'='*80}\n")

    # ========================================
    # 5. Create policy
    # ========================================
    print(f"[NaviSim] Available policies: {list_policies()}\n")
    print(f"[NaviSim] Creating policy: {args.policy}")

    # Prepare policy config based on policy type
    if args.policy in ["forward", "circle", "random"]:
        policy = get_policy(
            args.policy,
            num_envs=args.num_envs,
            observation_space=env.single_observation_space,
            action_space=env.single_action_space,
            device=env.device,
            action_scale=args.action_scale,
        )

    elif args.policy == "waypoint":
        config = WaypointPolicyConfig(
            name="waypoint",
            device=env.device,
            action_scale=args.action_scale,
            action_pattern=args.pattern,
        )
        policy = get_policy(
            args.policy,
            config=config,
            num_envs=args.num_envs,
            observation_space=env.single_observation_space,
            action_space=env.single_action_space,
        )

    elif args.policy in ["rsl_rl", "sb3"]:
        if not args.checkpoint:
            raise ValueError(f"--checkpoint required for {args.policy} policy")

        config = LearnedPolicyConfig(
            name=args.policy,
            checkpoint_path=args.checkpoint,
            device=env.device,
            deterministic=True,
        )
        policy = get_policy(
            args.policy,
            config=config,
            num_envs=args.num_envs,
            observation_space=env.single_observation_space,
            action_space=env.single_action_space,
        )

    else:
        policy = get_policy(
            args.policy,
            num_envs=args.num_envs,
            observation_space=env.single_observation_space,
            action_space=env.single_action_space,
            device=env.device,
        )

    print(f"[NaviSim] Policy created: {policy}\n")
    print(f"[NaviSim] Policy info:")
    for key, value in policy.get_info().items():
        print(f"    {key}: {value}")
    print()

    # ========================================
    # 6. Run simulation
    # ========================================
    obs, _ = env.reset()
    policy.reset()

    t = 0
    episode_rewards = torch.zeros(args.num_envs, device=env.device)
    episode_lengths = torch.zeros(args.num_envs, device=env.device)

    print(f"[NaviSim] Starting simulation...")
    print(f"  Max steps: {args.max_steps}\n")

    while simulation_app.is_running() and t < args.max_steps:

        # Get action from policy
        actions = policy.compute_action(obs)

        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Update policy (for online learning if supported)
        policy.update(obs, actions, rewards, terminated | truncated, info)

        # Track metrics
        episode_rewards += rewards
        episode_lengths += 1

        # Handle episode completion
        dones = terminated | truncated
        if dones.any():
            done_indices = torch.where(dones)[0]
            for idx in done_indices:
                print(f"[Step {t:4d}] Env {idx} completed | "
                      f"Reward: {episode_rewards[idx].item():+.3f} | "
                      f"Length: {int(episode_lengths[idx].item())}")

                episode_rewards[idx] = 0
                episode_lengths[idx] = 0

            # Reset policy for done environments
            policy.reset(done_indices)

        # Progress logging
        if t % 100 == 0 and t > 0:
            print(f"[Step {t:4d}] Running... | "
                  f"Mean reward: {rewards.mean().item():+.4f}")

        t += 1

    # ========================================
    # 7. Print final statistics
    # ========================================
    print(f"\n{'='*80}")
    print(f"[NaviSim] Simulation Complete")
    print(f"{'='*80}")
    print(f"  Total steps: {t}")
    print(f"  Policy: {args.policy}")
    print(f"  Policy statistics:")
    for key, value in policy.get_info().items():
        print(f"    {key}: {value}")
    print(f"{'='*80}\n")

    # ========================================
    # 8. Cleanup
    # ========================================
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
