# navisim_isaacsim/__init__.py

"""IsaacLab integration for NaviSim navigation environments."""

import os
import gymnasium as gym

# Import environment and config classes
from .tasks.navigation_env import NavisimNavigationEnv
from .configs.navigation_env_cfg import NavisimNavigationEnvCfg

# Import agents module for entry point resolution
from . import agents

# Register environment with Gymnasium including RL agent configurations
gym.register(
    id="Isaac-NavisimNavigation-Jetbot-v0",
    entry_point="navisim_isaacsim.tasks.navigation_env:NavisimNavigationEnv",
    disable_env_checker=True,  # IsaacLab handles its own validation
    kwargs={
        "env_cfg_entry_point": NavisimNavigationEnvCfg,
        # RSL-RL agent configuration (Python class)
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NavisimNavigationPPORunnerCfg",
        # Stable-Baselines3 agent configuration (YAML file)
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

__all__ = [
    "NavisimNavigationEnv",
    "NavisimNavigationEnvCfg",
    "agents",
]
