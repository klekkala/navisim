# scripts/compare_policies.py

"""Compare multiple policies on NaviSim environment."""

import argparse
import torch
from pathlib import Path
from datetime import datetime
import json
from isaaclab.app import AppLauncher

# ========================================
# 1. Parse arguments
# ========================================
parser = argparse.ArgumentParser(description="Compare multiple navigation policies")
parser.add_argument("--task", type=str, default="Isaac-NavisimNavigation-Jetbot-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments per policy")
parser.add_argument("--policies", type=str, nargs="+", default=["forward", "circle", "random"],
                   help="List of policy names to compare")
parser.add_argument("--checkpoints", type=str, nargs="*", default=[],
                   help="Checkpoints for learned policies (in order)")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes per policy")
parser.add_argument("--max_steps_per_episode", type=int, default=500, help="Max steps per episode")
parser.add_argument("--save_results", action="store_true", help="Save comparison results to file")
parser.add_argument("--results_dir", type=str, default="results/policy_comparison",
                   help="Directory to save results")

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
from policies.learned_policies import LearnedPolicyConfig


def evaluate_policy(env, policy, num_episodes, max_steps_per_episode):
    """
    Evaluate a single policy.

    Returns:
        Dictionary with evaluation metrics.
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    current_episode = 0
    current_episode_reward = torch.zeros(env.num_envs, device=env.device)
    current_episode_length = torch.zeros(env.num_envs, device=env.device)
    step_in_episode = 0

    obs, _ = env.reset()
    policy.reset()

    print(f"\n  Evaluating for {num_episodes} episodes...")

    while current_episode < num_episodes and simulation_app.is_running():

        # Get action from policy
        actions = policy.compute_action(obs)

        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Accumulate rewards
        current_episode_reward += rewards
        current_episode_length += 1
        step_in_episode += 1

        # Check for episode completion
        dones = terminated | truncated
        if dones.any() or step_in_episode >= max_steps_per_episode:
            # Record completed episodes
            for idx in range(env.num_envs):
                if dones[idx] or step_in_episode >= max_steps_per_episode:
                    episode_rewards.append(current_episode_reward[idx].item())
                    episode_lengths.append(current_episode_length[idx].item())

                    # Reset for this env
                    current_episode_reward[idx] = 0
                    current_episode_length[idx] = 0

                    current_episode += 1

                    if current_episode % 5 == 0:
                        print(f"    Completed {current_episode}/{num_episodes} episodes")

                    if current_episode >= num_episodes:
                        break

            # Reset environment
            if current_episode < num_episodes:
                obs, _ = env.reset()
                policy.reset()
                step_in_episode = 0

    # Compute statistics
    episode_rewards = torch.tensor(episode_rewards[:num_episodes])
    episode_lengths = torch.tensor(episode_lengths[:num_episodes])

    return {
        "mean_reward": episode_rewards.mean().item(),
        "std_reward": episode_rewards.std().item(),
        "min_reward": episode_rewards.min().item(),
        "max_reward": episode_rewards.max().item(),
        "mean_length": episode_lengths.mean().item(),
        "std_length": episode_lengths.std().item(),
        "num_episodes": num_episodes,
        "all_rewards": episode_rewards.tolist(),
        "all_lengths": episode_lengths.tolist(),
    }


def main():
    """Compare multiple policies."""

    # ========================================
    # 4. Create environment
    # ========================================
    env_cfg = NavisimNavigationEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = NavisimNavigationEnv(cfg=env_cfg, render_mode=None)  # Headless for speed

    print(f"\n{'='*80}")
    print(f"[NaviSim] Policy Comparison")
    print(f"{'='*80}")
    print(f"  Task: {args.task}")
    print(f"  Num envs: {args.num_envs}")
    print(f"  Num episodes per policy: {args.num_episodes}")
    print(f"  Max steps per episode: {args.max_steps_per_episode}")
    print(f"  Policies to compare: {args.policies}")
    print(f"{'='*80}\n")

    # ========================================
    # 5. Evaluate each policy
    # ========================================
    results = {}
    checkpoint_idx = 0

    for policy_name in args.policies:
        print(f"[Policy: {policy_name}]")
        print(f"{'-'*80}")

        # Create policy
        try:
            if policy_name in ["rsl_rl", "sb3"]:
                # Learned policy - needs checkpoint
                if checkpoint_idx >= len(args.checkpoints):
                    print(f"  ERROR: No checkpoint provided for {policy_name}")
                    continue

                config = LearnedPolicyConfig(
                    name=policy_name,
                    checkpoint_path=args.checkpoints[checkpoint_idx],
                    device=env.device,
                    deterministic=True,
                )
                policy = get_policy(
                    policy_name,
                    config=config,
                    num_envs=args.num_envs,
                    observation_space=env.single_observation_space,
                    action_space=env.single_action_space,
                )
                checkpoint_idx += 1

            else:
                # Scripted policy
                policy = get_policy(
                    policy_name,
                    num_envs=args.num_envs,
                    observation_space=env.single_observation_space,
                    action_space=env.single_action_space,
                    device=env.device,
                )

            print(f"  Policy created: {policy}")

            # Evaluate policy
            policy_results = evaluate_policy(
                env, policy, args.num_episodes, args.max_steps_per_episode
            )

            results[policy_name] = policy_results

            # Print results
            print(f"\n  Results:")
            print(f"    Mean reward: {policy_results['mean_reward']:+.3f} ± {policy_results['std_reward']:.3f}")
            print(f"    Min/Max reward: {policy_results['min_reward']:+.3f} / {policy_results['max_reward']:+.3f}")
            print(f"    Mean length: {policy_results['mean_length']:.1f} ± {policy_results['std_length']:.1f}")
            print(f"{'-'*80}\n")

        except Exception as e:
            print(f"  ERROR: Failed to evaluate {policy_name}: {e}\n")
            results[policy_name] = {"error": str(e)}

    # ========================================
    # 6. Print comparison summary
    # ========================================
    print(f"\n{'='*80}")
    print(f"[Comparison Summary]")
    print(f"{'='*80}\n")

    # Sort by mean reward
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    sorted_policies = sorted(valid_results.items(), key=lambda x: x[1]["mean_reward"], reverse=True)

    print(f"{'Rank':<6} {'Policy':<20} {'Mean Reward':<15} {'Mean Length':<15}")
    print(f"{'-'*80}")
    for rank, (policy_name, policy_results) in enumerate(sorted_policies, 1):
        print(f"{rank:<6} {policy_name:<20} "
              f"{policy_results['mean_reward']:+.3f} ± {policy_results['std_reward']:<.3f}  "
              f"{policy_results['mean_length']:.1f} ± {policy_results['std_length']:.1f}")

    print(f"{'='*80}\n")

    # ========================================
    # 7. Save results
    # ========================================
    if args.save_results:
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"comparison_{timestamp}.json"

        comparison_data = {
            "timestamp": timestamp,
            "task": args.task,
            "num_envs": args.num_envs,
            "num_episodes": args.num_episodes,
            "max_steps_per_episode": args.max_steps_per_episode,
            "policies": args.policies,
            "results": results,
        }

        with open(results_file, "w") as f:
            json.dump(comparison_data, f, indent=2)

        print(f"[NaviSim] Results saved to: {results_file}\n")

    # ========================================
    # 8. Cleanup
    # ========================================
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
