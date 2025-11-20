# scripts/run_with_direct_camera_access.py

"""Run Navisim navigation with direct USD camera access (no Isaac Lab Camera sensor)."""

import argparse
from isaaclab.app import AppLauncher

# ---------------------------------------------------
# 1. Parse arguments
# ---------------------------------------------------
parser = argparse.ArgumentParser(description="Run Navisim with direct camera access.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum simulation steps")
parser.add_argument("--save_interval", type=int, default=100, help="Steps between image saves")
parser.add_argument("--output_dir", type=str, default="outputs/jetbot_camera", help="Directory for images")

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
import omni.replicator.core as rep


def save_camera_image(rgb_array, output_dir: str, prefix: str = "camera"):
    """Save camera RGB array to image file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # Ensure uint8 format
    if rgb_array.dtype != np.uint8:
        if rgb_array.max() <= 1.0:
            rgb_array = (rgb_array * 255).astype(np.uint8)
        else:
            rgb_array = rgb_array.astype(np.uint8)

    # Save using PIL
    img = Image.fromarray(rgb_array)
    img.save(filepath)

    return filepath


def main():
    """Run navigation with direct camera rendering."""

    # ---------------------------------------------------
    # 4. Create standard environment (no camera sensor)
    # ---------------------------------------------------
    env_cfg = NavisimNavigationEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

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
    print(f"[Navisim] Navigation Environment with Direct Camera Access")
    print(f"{'='*80}")
    print(f"  Number of environments: {args.num_envs}")
    print(f"  Device: {env.device}")
    print(f"  Save interval: {args.save_interval} steps")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*80}\n")

    # ---------------------------------------------------
    # 6. Reset environment
    # ---------------------------------------------------
    obs, _ = env.reset()
    print(f"[Navisim] Environment reset complete")
    print(f"  Observation shape: {obs['policy'].shape}")

    # ---------------------------------------------------
    # 7. Setup render product for Jetbot camera (if it exists)
    # ---------------------------------------------------
    camera_prim_path = "/World/envs/env_0/Jetbot/chassis/rgb_camera/jetbot_camera"
    render_product = None
    rgb_annotator = None

    try:
        # Create render product from the camera prim
        render_product = rep.create.render_product(camera_prim_path, resolution=(640, 480))

        # Create RGB annotator
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([render_product])

        print(f"[Camera] Initialized Replicator for: {camera_prim_path}")
        camera_available = True
    except Exception as e:
        print(f"[Camera] Warning: Could not initialize camera rendering: {e}")
        print(f"[Camera] Continuing without camera capture...")
        camera_available = False

    # ---------------------------------------------------
    # 8. Main simulation loop
    # ---------------------------------------------------
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

        # Capture camera image at intervals
        if camera_available and t % args.save_interval == 0 and t > 0:
            try:
                # Get RGB data from annotator
                rgb_data = rgb_annotator.get_data()

                if rgb_data is not None:
                    # Convert to numpy array
                    rgb_array = np.array(rgb_data, dtype=np.uint8)

                    if rgb_array.size > 0 and rgb_array.max() > 0:
                        filepath = save_camera_image(
                            rgb_array,
                            output_dir=args.output_dir,
                            prefix=f"jetbot_cam_step{t:04d}"
                        )
                        print(f"[Step {t:4d}] Camera image saved to: {filepath}")
                    else:
                        print(f"[Step {t:4d}] Warning: Camera image is empty or black")
                else:
                    print(f"[Step {t:4d}] Warning: No camera data")
            except Exception as e:
                print(f"[Step {t:4d}] Camera capture error: {e}")

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
    if camera_available:
        print(f"  Camera images saved to: {args.output_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
