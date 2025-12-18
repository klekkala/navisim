# navisim_isaacsim/__init__.py

"""IsaacLab integration for NaviSim navigation environments."""

import os
import gymnasium as gym

# Import environment and config classes
from navisim_lab.envs.warehouse.warehouse_env import WarehouseEnv
from navisim_lab.envs.warehouse.warehouse_env_cfg import WarehouseEnvCfg

# Register environment with Gymnasium including RL agent configurations
gym.register(
    id="Isaac-NavisimNavigation-Jetbot-v0",
    entry_point="navisim_isaacsim.tasks.navigation_env:WarehouseEnv",
    disable_env_checker=True,  # IsaacLab handles its own validation
    kwargs={
        "env_cfg_entry_point": WarehouseEnvCfg,
    },
)

__all__ = [
    "WarehouseEnv",
    "WarehouseEnvCfg",
]
