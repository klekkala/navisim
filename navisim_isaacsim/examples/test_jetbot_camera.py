#!/usr/bin/env python3
"""Test script to verify Jetbot camera works and capture POV images.

This script:
1. Creates the warehouse environment with Jetbot
2. Verifies camera is in the scene
3. Runs random actions
4. Captures and saves camera images every N steps
5. Optionally displays live camera feed

Usage:
    # Save images only
    python examples/test_jetbot_camera.py --num_steps 100 --save_every 10

    # Display live feed (requires display)
    python examples/test_jetbot_camera.py --num_steps 100 --display

    # Headless mode with image saving
    python examples/test_jetbot_camera.py --num_steps 100 --save_every 10 --headless
"""

import argparse
import sys
from pathlib import Path

# Add navisim_isaacsim to path
SCRIPT_DIR = Path(__file__).resolve().parent
NAVISIM_ISAACSIM_DIR = SCRIPT_DIR.parent
if str(NAVISIM_ISAACSIM_DIR) not in sys.path:
    sys.path.insert(0, str(NAVISIM_ISAACSIM_DIR))

from isaaclab.app import AppLauncher

# Parse args before AppLauncher
parser = argparse.ArgumentParser(description="Test Jetbot camera")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--num_steps", type=int, default=100, help="Number of steps to run")
parser.add_argument("--save_every", type=int, default=0, help="Save images every N steps (0=don't save)")
parser.add_argument("--output_dir", type=str, default="outputs/jetbot_pov", help="Output directory for images")
parser.add_argument("--display", action="store_true", help="Display live camera feed")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import after AppLauncher
import gymnasium as gym
import torch
import navisim_lab.tasks
from navisim_lab.utils.camera_utils import (
    get_camera_images,
    save_camera_batch,
    visualize_camera_stream,
)

def main():
    print("="*60)
    print("Jetbot Camera Test")
    print("="*60)

    # Create environment
    print(f"\n1. Creating environment...")
    env = gym.make("Navisim-Warehouse-Jetbot-v0", num_envs=args.num_envs)
    print(f"   ✓ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Check if camera is in scene
    print(f"\n2. Checking for camera...")
    if "jetbot_camera" in env.unwrapped.scene:
        print(f"   ✓ Camera found in scene")
        camera = env.unwrapped.scene["jetbot_camera"]
        print(f"   Camera config:")
        print(f"     - Resolution: {camera.cfg.width}x{camera.cfg.height}")
        print(f"     - Data types: {camera.cfg.data_types}")
        print(f"     - Update pose: {camera.cfg.update_latest_camera_pose}")
    else:
        print(f"   ✗ Camera not found in scene!")
        print(f"   Available assets: {list(env.unwrapped.scene.keys())}")
        env.close()
        return 1

    # Reset environment
    print(f"\n3. Resetting environment...")
    obs, _ = env.reset()
    print(f"   ✓ Environment reset")

    # Test getting camera images
    print(f"\n4. Testing camera image capture...")
    rgb = get_camera_images(env.unwrapped, "jetbot_camera")
    if rgb is not None:
        print(f"   ✓ Camera images retrieved")
        print(f"   Shape: {rgb.shape}")
        print(f"   Dtype: {rgb.dtype}")
        print(f"   Range: [{rgb.min():.3f}, {rgb.max():.3f}]")
    else:
        print(f"   ✗ Failed to get camera images")
        env.close()
        return 1

    # Run simulation
    print(f"\n5. Running simulation for {args.num_steps} steps...")
    print(f"   Save images every {args.save_every} steps" if args.save_every > 0 else "   Not saving images")
    print(f"   Display live feed: {args.display}")

    output_dir = Path(args.output_dir)

    for step in range(args.num_steps):
        # Random actions
        actions = torch.randn(args.num_envs, env.unwrapped.cfg.num_actions, device=env.unwrapped.device)

        # Step environment
        obs, rewards, dones, truncated, info = env.step(actions)

        # Save images periodically
        if args.save_every > 0 and step % args.save_every == 0:
            save_camera_batch(
                env.unwrapped,
                output_dir,
                camera_name="jetbot_camera",
                prefix="jetbot_pov",
                step=step
            )

        # Display live feed
        if args.display:
            if not visualize_camera_stream(env.unwrapped, "jetbot_camera"):
                print("   Display window closed by user")
                break

        # Progress
        if step % 20 == 0:
            print(f"   Step {step}/{args.num_steps}")

    print(f"\n   ✓ Simulation complete!")

    # Final summary
    print(f"\n6. Summary:")
    print(f"   Steps completed: {step + 1}")
    if args.save_every > 0:
        num_saved = (step + 1) // args.save_every
        print(f"   Images saved: {num_saved} batches to {output_dir}")
    print(f"   Average reward: {rewards.mean().item():.3f}")

    # Cleanup
    env.close()
    simulation_app.close()

    print("\n" + "="*60)
    print("✓ Camera test complete!")
    print("="*60)

    return 0


if __name__ == "__main__":
    exit(main())
