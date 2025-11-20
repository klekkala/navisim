# scripts/run_simple_capture.py

"""Run Navisim navigation with simple viewport image capture."""

import argparse
from isaaclab.app import AppLauncher

# ---------------------------------------------------
# 1. Parse arguments
# ---------------------------------------------------
parser = argparse.ArgumentParser(description="Run Navisim navigation with viewport capture.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum simulation steps")
parser.add_argument("--save_interval", type=int, default=100, help="Steps between image saves")
parser.add_argument("--output_dir", type=str, default="outputs/images", help="Directory for images")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ---------------------------------------------------
# 2. Launch Isaac Sim
# ---------------------------------------------------
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------
# 3. Import modules AFTER launch
# ---------------------------------------------------
import sys
import os
from datetime import datetime
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.navigation_env_cfg import NavisimNavigationEnvCfg
from tasks.navigation_env import NavisimNavigationEnv
import torch
from PIL import Image
import omni.kit.app


def save_viewport_image(output_dir: str, prefix: str = "image"):
    """Save the current viewport as an image using Isaac Sim's screenshot API.

    Args:
        output_dir: Directory to save images
        prefix: Prefix for the filename

    Returns:
        str: Path to saved image file
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # Use Isaac Sim's built-in screenshot functionality
    # This directly captures the viewport framebuffer
    import omni.kit.capture.viewport

    capture_instance = omni.kit.capture.viewport.CaptureExtension.get_instance()
    if capture_instance:
        # Capture to file
        capture_instance.capture_viewport_to_file(filepath)
        return filepath

    return None


def main():
    """Run navigation with viewport capture."""

    # Create environment
    env_cfg = NavisimNavigationEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    if hasattr(args, 'device') and args.device:
        env_cfg.sim.device = args.device

    print(f"[Navisim] Using device: {env_cfg.sim.device}")

    # Create environment with human render mode (shows viewport)
    env = NavisimNavigationEnv(
        cfg=env_cfg,
        render_mode="human" if not args.headless else None
    )

    print(f"\n{'='*80}")
    print(f"[Navisim] Navigation Environment Created")
    print(f"{'='*80}")
    print(f"  Number of environments: {args.num_envs}")
    print(f"  Device: {env.device}")
    print(f"  Render mode: {'human' if not args.headless else 'None'}")
    print(f"  Save interval: {args.save_interval} steps")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*80}\n")

    # Reset environment
    obs, _ = env.reset()
    print(f"[Navisim] Environment reset complete")
    print(f"  Observation shape: {obs['policy'].shape}")

    # Check if viewport capture is available
    capture_available = False
    if not args.headless:
        try:
            import omni.kit.capture.viewport
            capture_instance = omni.kit.capture.viewport.CaptureExtension.get_instance()
            if capture_instance:
                capture_available = True
                print(f"[Capture] Viewport capture initialized")
        except Exception as e:
            print(f"[Capture] Warning: Could not initialize viewport capture: {e}")

    # Main simulation loop
    t = 0
    episode_rewards = torch.zeros(args.num_envs, device=env.device)

    print(f"\n[Navisim] Starting simulation loop...\n")

    while simulation_app.is_running() and t < args.max_steps:

        # Sample actions
        if t % 120 < 100:
            actions = env.sample_forward_action()
        else:
            actions = env.sample_turn_action()

        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)
        episode_rewards += rewards

        # Capture viewport at intervals
        if capture_available and t % args.save_interval == 0 and t > 0:
            try:
                filepath = save_viewport_image(
                    output_dir=args.output_dir,
                    prefix=f"jetbot_step{t:04d}"
                )
                if filepath:
                    print(f"[Step {t:4d}] Image saved to: {filepath}")
            except Exception as e:
                print(f"[Step {t:4d}] Capture error: {e}")

        # Episode completion
        dones = terminated | truncated
        if dones.any():
            done_indices = torch.where(dones)[0]
            for idx in done_indices:
                print(f"[Step {t:4d}] Env {idx} completed | Reward: {episode_rewards[idx].item():.3f}")
            episode_rewards[dones] = 0.0

        # Progress logging
        if t % 100 == 0 and t > 0:
            print(f"[Step {t:4d}] Mean reward: {rewards.mean().item():+.4f} | "
                  f"Cumulative: {episode_rewards.mean().item():.3f}")

        t += 1

    # Cleanup
    print(f"\n[Navisim] Simulation completed after {t} steps")
    print(f"  Final mean cumulative reward: {episode_rewards.mean().item():.3f}")
    if capture_available:
        print(f"  Images saved to: {args.output_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
