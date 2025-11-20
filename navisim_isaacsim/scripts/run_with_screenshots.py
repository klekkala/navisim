# scripts/run_with_screenshots.py

"""Run Navisim navigation environment with periodic screenshots of the viewer."""

import argparse
from isaaclab.app import AppLauncher

# ---------------------------------------------------
# 1. Parse arguments (DO NOT import Isaac modules yet)
# ---------------------------------------------------
parser = argparse.ArgumentParser(description="Run Navisim navigation with screenshots.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum simulation steps")
parser.add_argument("--screenshot_interval", type=int, default=100, help="Steps between screenshots")
parser.add_argument("--output_dir", type=str, default="outputs/screenshots", help="Directory for screenshots")

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
import sys
import os
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.navigation_env_cfg import NavisimNavigationEnvCfg
from tasks.navigation_env import NavisimNavigationEnv
import torch
import omni.kit.viewport.utility as vp_utils
from PIL import Image
import numpy as np


def save_screenshot(viewport_window, output_dir: str, prefix: str = "screenshot"):
    """Save a screenshot of the current viewer.

    Args:
        viewport_window: Viewport window to capture
        output_dir: Directory to save screenshots
        prefix: Prefix for the filename

    Returns:
        str: Path to saved screenshot file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # Get viewport texture
    texture = viewport_window.get_texture()
    if texture is not None:
        # Get image data as numpy array
        image_data = vp_utils.get_texture_numpy(texture, return_srgb=True)

        if image_data is not None:
            # Convert RGBA to RGB if needed
            if image_data.shape[-1] == 4:
                image_data = image_data[:, :, :3]

            # Save using PIL
            img = Image.fromarray(image_data.astype(np.uint8))
            img.save(filepath)

            return filepath

    return None


def main():
    """Run navigation task with periodic screenshots."""

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
    print(f"  Screenshot interval: {args.screenshot_interval} steps")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*80}\n")

    # ---------------------------------------------------
    # 6. Reset environment
    # ---------------------------------------------------
    obs, _ = env.reset()
    print(f"[Navisim] Environment reset complete")
    print(f"  Observation shape: {obs['policy'].shape}")

    # Get viewport window for screenshots (only in non-headless mode)
    viewport_window = None
    if not args.headless:
        try:
            import omni.kit.viewport.utility as vp_utils
            viewport_window = vp_utils.get_active_viewport_window()
            print(f"[Screenshot] Viewport initialized for capture")
        except Exception as e:
            print(f"[Screenshot] Warning: Could not initialize viewport: {e}")
            print(f"[Screenshot] Continuing without screenshots...")

    # ---------------------------------------------------
    # 7. Main simulation loop with screenshots
    # ---------------------------------------------------
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

        # Capture screenshot at intervals
        if viewport_window is not None and t % args.screenshot_interval == 0:
            try:
                filepath = save_screenshot(
                    viewport_window,
                    output_dir=args.output_dir,
                    prefix=f"jetbot_step{t:04d}"
                )
                if filepath:
                    print(f"[Screenshot] Saved to: {filepath}")
                else:
                    print(f"[Screenshot] Failed to capture at step {t}")
            except Exception as e:
                print(f"[Screenshot] Error at step {t}: {e}")

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

    if not args.headless:
        print(f"  Screenshots saved to: {args.output_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
