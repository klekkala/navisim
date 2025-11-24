# scripts/run_with_jetbot_camera.py

"""Run Navisim navigation with Jetbot first-person camera capture using Isaac Lab Camera sensor.

This script:
- Runs the navigation task with the Jetbot embodiment
- Captures images from the Jetbot's onboard camera using Isaac Lab Camera API
- Shows Isaac Sim GUI by default (use --headless to disable)
- Saves camera images to outputs/jetbot_pov/ at specified intervals for debugging
- Updates camera every simulation step to capture latest images

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
    # 7. Setup camera capture using Isaac Lab Camera sensor (BEFORE reset)
    # ---------------------------------------------------
    # IMPORTANT: Camera must be created BEFORE env.reset() so it gets initialized
    # Correct camera path discovered from scene inspection
    camera_prim_path = "/World/envs/env_0/Jetbot/chassis/rgb_camera/jetbot_camera"
    camera = None
    camera_available = False

    try:
        print(f"[Camera] Setting up Isaac Lab Camera sensor...")
        print(f"[Camera]   - Camera path: {camera_prim_path}")

        # Create camera configuration for the existing camera prim
        camera_cfg = CameraCfg(
            prim_path=camera_prim_path,
            update_period=0,  # Update every step
            height=480,
            width=640,
            data_types=["rgb"],  # We want RGB images
            spawn=None,  # Don't spawn - use existing camera prim
        )

        # Create camera sensor
        camera = Camera(camera_cfg)

        # IMPORTANT: Stop and play simulation to trigger camera initialization callback
        # The camera's _initialize_callback is only called on PLAY events
        # Since the sim was already playing when we created the camera, we need to trigger a new PLAY event
        print(f"[Camera] Triggering simulation play to initialize camera...")
        sim.pause()
        sim.play()

        print(f"[Camera] ✓ Successfully initialized Isaac Lab Camera")
        print(f"[Camera]   - Resolution: 640x480")
        print(f"[Camera]   - Data types: rgb")
        print(f"[Camera]   - Is initialized: {camera.is_initialized}")
        camera_available = True

    except Exception as e:
        import traceback
        print(f"[Camera] ✗ Failed to initialize camera: {e}")
        print(f"[Camera] Traceback:")
        traceback.print_exc()
        print(f"[Camera] Continuing without camera capture...")
        camera_available = False

    # ---------------------------------------------------
    # 8. Reset environment
    # ---------------------------------------------------
    obs, _ = env.reset()
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

    print(f"\n[Navisim] Starting simulation loop...")
    print(f"  Capturing Jetbot's first-person view\n")

    while simulation_app.is_running() and t < args.max_steps:

        # Sample actions
        if t % 120 < 100:
            actions = env.sample_forward_action()
        else:
            actions = env.sample_turn_action()

        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)
        episode_rewards += rewards

        # Update camera sensor (call every step to get latest data)
        if camera_available:
            camera.update(dt=env.sim.cfg.dt)

        # Capture camera image at intervals for debugging
        if camera_available and t % args.save_interval == 0 and t > 0:
            try:
                # Get RGB data from Isaac Lab Camera
                # camera.data.output is a dict: {"rgb": tensor of shape (1, H, W, 3)}
                if "rgb" in camera.data.output:
                    rgb_tensor = camera.data.output["rgb"]  # Shape: (1, H, W, 3)

                    # Convert tensor to numpy array
                    rgb_array = rgb_tensor[0].cpu().numpy()  # Shape: (H, W, 3)

                    # Check if image has valid data
                    if rgb_array.size > 0 and rgb_array.max() > 0:
                        filepath = save_camera_image(
                            rgb_array,
                            output_dir=args.output_dir,
                            prefix=f"jetbot_pov_step{t:04d}"
                        )
                        print(f"[Step {t:4d}] ✓ Jetbot POV saved to: {filepath}")
                        print(f"           Image shape: {rgb_array.shape}, dtype: {rgb_array.dtype}, range: [{rgb_array.min():.2f}, {rgb_array.max():.2f}]")
                    else:
                        print(f"[Step {t:4d}] ✗ Warning: Camera image is empty or black")
                        print(f"           Image shape: {rgb_array.shape}, dtype: {rgb_array.dtype}, range: [{rgb_array.min():.2f}, {rgb_array.max():.2f}]")
                else:
                    print(f"[Step {t:4d}] ✗ Warning: No RGB data in camera output")
                    print(f"           Available keys: {list(camera.data.output.keys())}")
            except Exception as e:
                import traceback
                print(f"[Step {t:4d}] ✗ Camera capture error: {e}")
                traceback.print_exc()

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
        print(f"  Jetbot POV images saved to: {args.output_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
