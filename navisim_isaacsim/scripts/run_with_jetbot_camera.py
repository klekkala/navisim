# scripts/run_with_jetbot_camera.py

"""Run Navisim navigation with Jetbot first-person camera capture using Isaac Lab Camera sensor.

This script demonstrates proper camera tracking in Isaac Lab by:
- Referencing the existing camera in Jetbot USD (spawn=None)
- Enabling update_latest_camera_pose=True for automatic transform tracking via XFormPrimView
- Capturing RGB images from the Jetbot's perspective as it moves through the warehouse
- Saving images at specified intervals for debugging/training data collection

Key Isaac Lab pattern:
    Camera is configured in scene config with:
    - spawn=None (reference existing camera prim, don't create new one)
    - update_latest_camera_pose=True (track transforms automatically)

    This allows Isaac Lab's XFormPrimView to track the camera's world position
    as the Jetbot moves, ensuring images update correctly.

Usage:
    # With GUI (default)
    ./run_isaac_lab.sh "python scripts/run_with_jetbot_camera.py --save_interval 100"

    # Headless mode
    ./run_isaac_lab.sh "python scripts/run_with_jetbot_camera.py --headless --save_interval 100"
"""

import argparse
from isaaclab.app import AppLauncher

# ---------------------------------------------------
# 1. Parse arguments FIRST (before launching app)
# ---------------------------------------------------
parser = argparse.ArgumentParser(description="Run Navisim with Jetbot first-person camera.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum simulation steps")
parser.add_argument("--save_interval", type=int, default=100, help="Steps between image saves")
parser.add_argument("--output_dir", type=str, default="outputs/jetbot_pov", help="Directory for images")

# AppLauncher adds --headless and other Isaac Lab args
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ---------------------------------------------------
# 2. Launch Isaac Sim via AppLauncher (sets up paths correctly)
# ---------------------------------------------------
# IMPORTANT: Enable cameras for Isaac Lab Camera sensor
args.enable_cameras = True  # Set this before creating AppLauncher

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------
# 3. Import modules AFTER SimulationApp launch
# ---------------------------------------------------
import sys
import os
from datetime import datetime
import numpy as np
import torch
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.navigation_env_cfg import NavisimNavigationEnvCfg
from tasks.navigation_env import NavisimNavigationEnv

# Isaac Lab camera sensor
# Camera imports - we'll use Isaac Lab's Camera but manually update position
from isaaclab.sensors.camera import Camera, CameraCfg


def save_camera_image(rgb_array, output_dir: str, prefix: str = "camera"):
    """Save camera RGB array to image file.

    Args:
        rgb_array: RGB image data as numpy array (H, W, 3)
        output_dir: Directory to save images
        prefix: Prefix for the filename

    Returns:
        str: Path to saved image file
    """
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
    """Run navigation with Jetbot first-person camera capture."""

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
    # Determine render mode based on headless argument
    if hasattr(args, 'headless') and args.headless:
        render_mode = None  # Headless mode
    else:
        render_mode = "human"  # GUI mode

    env = NavisimNavigationEnv(
        cfg=env_cfg,
        render_mode=render_mode
    )

    print(f"\n{'='*80}")
    print(f"[Navisim] Navigation Environment with Jetbot First-Person Camera")
    print(f"{'='*80}")
    print(f"  Number of environments: {args.num_envs}")
    print(f"  Device: {env.device}")
    print(f"  Render mode: {'Headless' if render_mode is None else 'GUI (human)'}")
    print(f"  Save interval: {args.save_interval} steps")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*80}\n")

    # ---------------------------------------------------
    # 6. Get simulation context and configure rendering
    # ---------------------------------------------------
    # Access the simulation context from the environment
    sim = env.sim

    print(f"[Rendering] Using Isaac Lab SimulationContext")

    # ---------------------------------------------------
    # 7. Reset environment (this initializes the scene including camera)
    # ---------------------------------------------------
    obs, _ = env.reset()

    # ---------------------------------------------------
    # 8. Access camera sensor from scene (with update_latest_camera_pose=True)
    # ---------------------------------------------------
    # The camera is now configured in the scene with update_latest_camera_pose=True
    # This enables XFormPrimView to track camera transforms as Jetbot moves
    camera = env.scene["jetbot_camera"]
    camera_available = True

    print(f"[Camera] ✓ Camera sensor accessed from scene")
    print(f"[Camera]   Path: {camera.cfg.prim_path}")
    print(f"[Camera]   Resolution: {camera.image_shape}")
    print(f"[Camera]   Update pose tracking: {camera.cfg.update_latest_camera_pose}")
    print(f"[Navisim] Environment reset complete")
    print(f"  Observation shape: {obs['policy'].shape}")

    # Debug: Verify scene entities are loaded
    print(f"\n[Scene] Loaded entities:")
    for entity_name in env.scene.keys():
        print(f"  - {entity_name}")

    # Check if Jetbot is present (check if key exists in scene.keys())
    scene_entities = list(env.scene.keys())
    if "jetbot" in scene_entities:
        jetbot_pos = env.scene["jetbot"].data.root_pos_w[0].cpu().numpy()
        print(f"\n[Jetbot] Position: {jetbot_pos}")
        print(f"[Jetbot] Number of instances: {env.scene['jetbot'].num_instances}")
    else:
        print(f"\n[ERROR] Jetbot not found in scene!")

    # ---------------------------------------------------
    # 8.5. Set viewport camera to look at the scene
    # ---------------------------------------------------
    if not args.headless:
        try:
            # Get Jetbot position to center the camera on it
            if "jetbot" in scene_entities:
                jetbot_pos = env.scene["jetbot"].data.root_pos_w[0].cpu().numpy()

                # Set camera to look at the Jetbot from a good angle
                # Position: above and behind the robot
                eye = jetbot_pos + np.array([3.0, 3.0, 2.0])  # 3m back, 3m right, 2m up
                target = jetbot_pos  # Look at the robot

                # Use SimulationContext's set_camera_view method (Isaac Lab way)
                sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
                print(f"\n[Viewport] ✓ Camera positioned to view Jetbot at: {jetbot_pos}")
            else:
                print(f"\n[Viewport] Cannot position camera - Jetbot not found")
        except Exception as e:
            print(f"\n[Viewport] Warning: Could not set camera view: {e}")

    # ---------------------------------------------------
    # 9. Main simulation loop with camera capture
    # ---------------------------------------------------
    t = 0
    episode_rewards = torch.zeros(args.num_envs, device=env.device)

    print(f"\n[Navisim] Starting simulation loop...", flush=True)
    print(f"  Max steps: {args.max_steps}", flush=True)
    print(f"  simulation_app.is_running(): {simulation_app.is_running()}", flush=True)
    print(f"  Capturing Jetbot's first-person view\n", flush=True)

    while simulation_app.is_running() and t < args.max_steps:

        # Sample actions
        if t % 120 < 100:
            actions = env.sample_forward_action()
        else:
            actions = env.sample_turn_action()

        # Step environment (this will apply actions, step physics, and update everything)
        # The env.step() method calls _pre_physics_step() which applies actions via set_joint_velocity_target()
        obs, rewards, terminated, truncated, info = env.step(actions)
        episode_rewards += rewards

        # Capture and save camera images at specified intervals
        if camera_available and t % args.save_interval == 0:
            # Update camera data (this calls update() internally which refreshes transforms via XFormPrimView)
            camera_data = camera.data

            # Get RGB image from first environment
            # camera.data.output["rgb"] shape: (num_envs, height, width, channels)
            rgb_tensor = camera_data.output["rgb"][0]  # Get first env's image

            # Convert to numpy and save
            rgb_array = rgb_tensor.cpu().numpy()

            # Get camera position for logging
            cam_pos = camera_data.pos_w[0].cpu().numpy()

            # Save image
            filepath = save_camera_image(rgb_array, args.output_dir, prefix="jetbot_pov")
            print(f"[Step {t:4d}] 📸 Saved camera image: {filepath}")
            print(f"           Camera position: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]")

        # Episode completion
        dones = terminated | truncated
        if dones.any():
            done_indices = torch.where(dones)[0]
            for idx in done_indices:
                print(f"[Step {t:4d}] Env {idx} completed | Reward: {episode_rewards[idx].item():.3f}")
            episode_rewards[dones] = 0.0

        # Progress logging with position and action debug
        if t % 50 == 0:
            jetbot_pos = env.scene["jetbot"].data.root_pos_w[0].cpu().numpy()
            jetbot_vel = env.scene["jetbot"].data.root_lin_vel_w[0].cpu().numpy()
            action_type = "FORWARD" if t % 120 < 100 else "TURN"

            print(f"[Step {t:4d}] {action_type} | Jetbot Pos: [{jetbot_pos[0]:.2f}, {jetbot_pos[1]:.2f}, {jetbot_pos[2]:.2f}] | "
                  f"Vel: [{jetbot_vel[0]:.2f}, {jetbot_vel[1]:.2f}, {jetbot_vel[2]:.2f}] | "
                  f"Reward: {rewards.mean().item():+.4f}", flush=True)

        t += 1

    # Cleanup
    print(f"\n[Navisim] Simulation completed after {t} steps")
    print(f"  Final mean cumulative reward: {episode_rewards.mean().item():.3f}")
    if camera_available:
        num_images_saved = (t // args.save_interval) + 1
        print(f"  Jetbot POV images saved: {num_images_saved} images in {args.output_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
