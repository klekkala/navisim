from isaaclab.app import AppLauncher

# 1) Launch Isaac Sim (Kit) BEFORE importing anything that touches omni/physx
app_launcher = AppLauncher(headless=False)  # set True for headless
simulation_app = app_launcher.app

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# 2) Now it's safe to import gym + your task registration
# Launch Kit first so omni.* modules exist
app = AppLauncher(headless=False).app

import gymnasium as gym
import navisim_lab_tasks  # registers task id
from isaaclab_tasks.utils import parse_env_cfg  # latest namespace :contentReference[oaicite:1]{index=1}

task_id = "Navisim-Warehouse-Jetbot-v0"
env_cfg = parse_env_cfg(task_id, num_envs=1)  # loads env_cfg_entry_point :contentReference[oaicite:2]{index=2}

env = gym.make(task_id, cfg=env_cfg)  # IMPORTANT: cfg must be passed
obs, _ = env.reset()

for _ in range(10):
    a = env.action_space.sample()
    obs, rew, term, trunc, info = env.step(a)

env.close()
app.close()