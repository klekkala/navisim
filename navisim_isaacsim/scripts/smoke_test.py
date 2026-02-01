from isaaclab.app import AppLauncher
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse arguments
parser = argparse.ArgumentParser(description="Smoke test for Navisim warehouse environment")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
# AppLauncher adds its own arguments (including --headless)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Enable cameras (required for environments with camera sensors)
args.enable_cameras = True

# 1) Launch Isaac Sim (Kit) BEFORE importing anything that touches omni/physx
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2) Now it's safe to import gym + your task registration
logger.info("Registering tasks...")

import gymnasium as gym
import torch
import navisim_lab.tasks  # registers task id
from isaaclab_tasks.utils import parse_env_cfg  # latest namespace

logger.info("Creating environment...")

task_id = "Navisim-Warehouse-Jetbot-v0"
env_cfg = parse_env_cfg(task_id, num_envs=args.num_envs)  # loads env_cfg_entry_point

env = gym.make(task_id, cfg=env_cfg)  # IMPORTANT: cfg must be passed
obs, _ = env.reset()

# Get the device from the unwrapped environment
device = env.unwrapped.device

logger.info("Running smoke test...")

for _ in range(10):
    # Sample action and convert to torch tensor on the correct device
    a = env.action_space.sample()
    a = torch.tensor(a, device=device, dtype=torch.float32)
    obs, rew, term, trunc, info = env.step(a)
    logger.info(f"Step reward: {obs}, {rew}, {term}")

env.close()
simulation_app.close()


logger.info("Finished smoke test...")
