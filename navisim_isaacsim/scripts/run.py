# scripts/run.py

"""Run Navisim navigation environment with Isaac Lab."""

import argparse
from isaaclab.app import AppLauncher

# ---------------------------------------------------
# 1. Parse arguments (DO NOT import Isaac modules yet)
# ---------------------------------------------------
parser = argparse.ArgumentParser(description="Run Navisim navigation with Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum simulation steps")

# AppLauncher injects: --headless, --enable_cameras, --device, etc.
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ---------------------------------------------------
# 2. Launch Isaac Sim (MUST happen before importing Isaac modules)
# ---------------------------------------------------
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------
# 3. Import Isaac Lab and Navisim modules AFTER launch
# ---------------------------------------------------
from configs.navigation_env_cfg import NavisimNavigationEnvCfg
from tasks.navigation_env import NavisimNavigationEnv


def main():
    """Run navigation task with DirectRLEnv."""

    # ---------------------------------------------------
    # 4. Create environment configuration
    # ---------------------------------------------------
    env_cfg = NavisimNavigationEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Override device if specified via command line
    if hasattr(args, 'device') and args.device:
        env_cfg.sim.device = args.device

    print(f"[Navisim] Using device: {env_cfg.sim.device}")

    # ---------------------------------------------------
    # 5. Create environment
    # ---------------------------------------------------
    env = NavisimNavigationEnv(
        cfg=env_cfg,
        render_mode="human" if not args.headless else None
    )

    print(f"\n{'='*80}")
    print(f"[Navisim] Navigation Environment Created")
    print(f"{'='*80}")
    print(f"  Number of environments: {args.num_envs}")
    print(f"  Observation space: {env.single_observation_space}")
    print(f"  Action space: {env.single_action_space}")
    print(f"  Device: {env.device}")
    print(f"  Headless mode: {args.headless}")
    print(f"{'='*80}\n")

    # ---------------------------------------------------
    # 6. Reset environment
    # ---------------------------------------------------
    obs, _ = env.reset()
    print(f"[Navisim] Environment reset complete")
    print(f"  Observation shape: {obs['policy'].shape}")

    # ---------------------------------------------------
    # 7. Main simulation loop
    # ---------------------------------------------------
    import torch  # Import here after Isaac Sim is initialized

    t = 0
    episode_rewards = torch.zeros(args.num_envs, device=env.device)

    print(f"\n[Navisim] Starting simulation loop...")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Control pattern: Forward (100 steps) → Turn (20 steps)\n")

    while simulation_app.is_running() and t < args.max_steps:

        # Sample actions: forward for 100 steps, turn for 20 steps
        if t % 120 < 100:
            actions = env.sample_forward_action()
        else:
            actions = env.sample_turn_action()

        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Accumulate rewards
        episode_rewards += rewards

        # Check for episode completion
        dones = terminated | truncated
        if dones.any():
            done_indices = torch.where(dones)[0]
            for idx in done_indices:
                print(f"[Step {t:4d}] Environment {idx} completed | "
                      f"Episode reward: {episode_rewards[idx].item():.3f}")
            episode_rewards[dones] = 0.0

        # Progress logging
        if t % 100 == 0 and t > 0:
            print(f"[Step {t:4d}] Mean reward (last step): {rewards.mean().item():+.4f} | "
                  f"Cumulative reward: {episode_rewards.mean().item():.3f}")

        t += 1

    # ---------------------------------------------------
    # 8. Cleanup
    # ---------------------------------------------------
    print(f"\n[Navisim] Simulation completed after {t} steps")
    print(f"  Final mean cumulative reward: {episode_rewards.mean().item():.3f}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
