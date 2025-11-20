# scripts/run_with_jetbot_camera.py

"""Run Navisim navigation with Jetbot first-person camera capture."""

import argparse
from isaaclab.app import AppLauncher

# ---------------------------------------------------
# 1. Parse arguments
# ---------------------------------------------------
parser = argparse.ArgumentParser(description="Run Navisim with Jetbot first-person camera.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum simulation steps")
parser.add_argument("--save_interval", type=int, default=100, help="Steps between image saves")
parser.add_argument("--output_dir", type=str, default="outputs/jetbot_pov", help="Directory for images")

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
from scene.warehouse_scene_with_jetbot_camera_cfg import NavisimWarehouseSceneWithJetbotCameraCfg
from tasks.navigation_env import NavisimNavigationEnv
import torch
from PIL import Image


def save_camera_image(rgb_tensor, output_dir: str, prefix: str = "camera"):
    """Save camera RGB tensor to image file.

    Args:
        rgb_tensor: RGB image data as torch tensor (H, W, 3) or numpy array
        output_dir: Directory to save images
        prefix: Prefix for the filename

    Returns:
        str: Path to saved image file
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # Convert to numpy if needed
    if torch.is_tensor(rgb_tensor):
        rgb_array = rgb_tensor.cpu().numpy()
    else:
        rgb_array = rgb_tensor

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
    # 4. Create environment with Jetbot camera scene
    # ---------------------------------------------------
    env_cfg = NavisimNavigationEnvCfg()

    # Replace scene with Jetbot camera-enabled version
    env_cfg.scene = NavisimWarehouseSceneWithJetbotCameraCfg(
        num_envs=args.num_envs,
        env_spacing=4.0,
    )

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
    print(f"[Navisim] Navigation Environment with Jetbot First-Person Camera")
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

    # Check camera availability
    camera_available = "jetbot_camera" in env.scene
    if camera_available:
        print(f"[Camera] Jetbot first-person camera initialized")
    else:
        print(f"[Camera] ERROR: Camera not found in scene!")
        return

    # ---------------------------------------------------
    # 7. Main simulation loop with camera capture
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

        # Update camera position to follow Jetbot (first environment)
        if camera_available:
            # Get Jetbot's current pose
            jetbot_pos = env.scene["jetbot"].data.root_pos_w[0]  # World position
            jetbot_quat = env.scene["jetbot"].data.root_quat_w[0]  # World orientation (w, x, y, z)

            # Camera offset in robot's local frame: 20cm forward, 10cm up
            import torch
            from isaaclab.utils.math import quat_rotate_inverse, quat_mul

            # Offset in robot's local frame
            local_offset = torch.tensor([0.2, 0.0, 0.1], device=env.device)

            # Rotate offset to world frame
            world_offset = quat_rotate_inverse(jetbot_quat, local_offset)

            # Set camera position
            camera_pos = jetbot_pos + world_offset

            # Camera orientation matches robot orientation
            camera_quat = jetbot_quat

            # Update camera transform via USD
            from pxr import Gf, UsdGeom
            camera_prim_path = f"/World/envs/env_0/JetbotCamera"
            camera_prim = env.scene._stage.GetPrimAtPath(camera_prim_path)

            if camera_prim:
                xform = UsdGeom.Xformable(camera_prim)
                xform.ClearXformOpOrder()

                # Set translation
                translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item()))

                # Set rotation (convert quaternion to rotation)
                orient_op = xform.AddOrientOp()
                # USD uses (w, x, y, z) like Isaac Lab
                orient_op.Set(Gf.Quatd(camera_quat[0].item(), camera_quat[1].item(), camera_quat[2].item(), camera_quat[3].item()))

        # Capture camera image at intervals
        if t % args.save_interval == 0 and t > 0:
            try:
                # Get RGB data from Jetbot's camera (first environment)
                camera_data = env.scene["jetbot_camera"].data.output["rgb"][0]

                if camera_data is not None:
                    filepath = save_camera_image(
                        camera_data,
                        output_dir=args.output_dir,
                        prefix=f"jetbot_pov_step{t:04d}"
                    )
                    print(f"[Step {t:4d}] Jetbot POV saved to: {filepath}")
                else:
                    print(f"[Step {t:4d}] Camera data is None")
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
    print(f"  Jetbot POV images saved to: {args.output_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
