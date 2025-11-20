# navisim_isaacsim/__init__.py

"""IsaacLab integration for NaviSim navigation environments."""

import gymnasium as gym

# Import environment and config classes
from .tasks.navigation_env import NavisimNavigationEnv
from .configs.navigation_env_cfg import NavisimNavigationEnvCfg

# Register environment with Gymnasium
gym.register(
    id="Isaac-NavisimNavigation-Jetbot-v0",
    entry_point="navisim_isaacsim.tasks.navigation_env:NavisimNavigationEnv",
    disable_env_checker=True,  # IsaacLab handles its own validation
)

__all__ = [
    "NavisimNavigationEnv",
    "NavisimNavigationEnvCfg",
]
