# agents/__init__.py

"""RL agent configurations for NaviSim navigation environments."""

import os
from pathlib import Path

# Get the directory path for agent configs
AGENTS_DIR = Path(__file__).parent

# Import agent configurations
from .rsl_rl_ppo_cfg import NavisimNavigationPPORunnerCfg

__all__ = [
    "NavisimNavigationPPORunnerCfg",
    "AGENTS_DIR",
]
