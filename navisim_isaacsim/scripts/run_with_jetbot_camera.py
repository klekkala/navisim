# scripts/run_with_jetbot_camera.py

"""Run Navisim navigation with Jetbot first-person camera capture.

Works with both the warehouse and outdoor (3DGS) environments.

Usage:
    # Warehouse environment (default)
    python scripts/run_with_jetbot_camera.py

    # Outdoor 3DGS environment
    python scripts/run_with_jetbot_camera.py --task Navisim-Outdoor-Jetbot-v0

    # Headless with custom output dir
    python scripts/run_with_jetbot_camera.py --task Navisim-Outdoor-Jetbot-v0 --headless --save_interval 50
"""

import argparse
from isaaclab.app import AppLauncher

# ---------------------------------------------------
# 1. Parse arguments FIRST (before launching app)
# ---------------------------------------------------
parser = argparse.ArgumentParser(description="Run Navisim with Jetbot first-person camera.")
parser.add_argument("--task", type=str, default="Navisim-Warehouse-Jetbot-v0",
                    choices=[
                        "Navisim-Warehouse-Jetbot-v0",
                        "Navisim-Outdoor-Jetbot-Camera-v0",
                    ],
                    help="Which environment to run (outdoor variant includes camera)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum simulation steps")
parser.add_argument("--save_interval", type=int, default=100, help="Steps between image saves")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Directory for saved images (defaults to ../outputs/<task_name>)")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------
# 2. Import modules AFTER SimulationApp launch
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

import navisim_lab.tasks  # registers all Gymnasium environments

from navisim_lab.envs.warehouse.warehouse_env import WarehouseEnv
from navisim_lab.envs.warehouse.warehouse_env_cfg import WarehouseEnvCfg
from navisim_lab.envs.outdoor.outdoor_env_with_camera_cfg import OutdoorEnvWithCameraCfg


def save_camera_image(rgb_array, output_dir: str, prefix: str = "camera"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filepath = os.path.join(output_dir, f"{prefix}_{timestamp}.png")
    if rgb_array.dtype != np.uint8:
        rgb_array = (rgb_array * 255).astype(np.uint8) if rgb_array.max() <= 1.0 else rgb_array.astype(np.uint8)
    Image.fromarray(rgb_array).save(filepath)
    return filepath


def main():
    # Select config based on task
    if args.task == "Navisim-Outdoor-Jetbot-Camera-v0":
        env_cfg = OutdoorEnvWithCameraCfg()
    else:
        env_cfg = WarehouseEnvCfg()

    env_cfg.scene.num_envs = args.num_envs
    if hasattr(args, 'device') and args.device:
        env_cfg.sim.device = args.device

    output_dir = args.output_dir or os.path.join(
        project_root, "outputs", args.task.lower().replace("-", "_")
    )

    render_mode = None if (hasattr(args, 'headless') and args.headless) else "human"
    env = WarehouseEnv(cfg=env_cfg, render_mode=render_mode)

    print(f"\n{'='*70}")
    print(f"  Task:        {args.task}")
    print(f"  Num envs:    {args.num_envs}")
    print(f"  Device:      {env.device}")
    print(f"  Output dir:  {output_dir}")
    print(f"{'='*70}\n")

    obs, _ = env.reset()

    camera = env.scene["jetbot_camera"]
    print(f"[Camera] path={camera.cfg.prim_path}  resolution={camera.image_shape}")

    print(f"[Scene] entities: {list(env.scene.keys())}")
    jetbot_pos = env.scene["jetbot"].data.root_pos_w[0].cpu().numpy()
    print(f"[Jetbot] spawn position: {jetbot_pos}")

    if not args.headless:
        try:
            eye = (jetbot_pos + np.array([3.0, -3.0, 2.0])).tolist()
            env.sim.set_camera_view(eye=eye, target=jetbot_pos.tolist())
        except Exception as e:
            print(f"[Viewport] Warning: {e}")

    t = 0
    episode_rewards = torch.zeros(args.num_envs, device=env.device)

    print(f"[Navisim] Starting loop (max {args.max_steps} steps)...\n")

    while simulation_app.is_running() and t < args.max_steps:
        # Alternate forward / turn every 120 steps
        actions = env.sample_forward_action() if t % 120 < 100 else env.sample_turn_action()
        obs, rewards, terminated, truncated, info = env.step(actions)
        episode_rewards += rewards

        # Capture POV image
        if t % args.save_interval == 0:
            rgb_tensor = camera.data.output["rgb"][0]
            rgb_array = rgb_tensor.cpu().numpy()
            cam_pos = camera.data.pos_w[0].cpu().numpy()
            filepath = save_camera_image(rgb_array, output_dir, prefix="jetbot_pov")
            print(f"[Step {t:4d}] Saved {filepath}  cam_pos=[{cam_pos[0]:.2f},{cam_pos[1]:.2f},{cam_pos[2]:.2f}]")

        dones = terminated | truncated
        if dones.any():
            episode_rewards[dones] = 0.0

        if t % 50 == 0:
            pos = env.scene["jetbot"].data.root_pos_w[0].cpu().numpy()
            vel = env.scene["jetbot"].data.root_lin_vel_w[0].cpu().numpy()
            motion = "FWD" if t % 120 < 100 else "TURN"
            print(f"[Step {t:4d}] {motion}  pos=[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]  "
                  f"vel=[{vel[0]:.2f},{vel[1]:.2f}]  reward={rewards.mean().item():+.4f}", flush=True)

        t += 1

    print(f"\n[Navisim] Done after {t} steps  |  mean reward: {episode_rewards.mean().item():.3f}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
