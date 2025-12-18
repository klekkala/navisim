import gymnasium as gym

# Register Gym environment ID -> Isaac Lab environment class
# Isaac Lab recommends using gym.register with env_cfg_entry_point in kwargs. :contentReference[oaicite:1]{index=1}
# navisim_lab_tasks/warehouse/__init__.py
gym.register(
    id="Navisim-Warehouse-Jetbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "navisim_lab.envs.warehouse.warehouse_cfg:WarehouseJetbotEnvCfg",
        # optional:
        "rsl_rl_cfg_entry_point": "navisim_lab.configs.rsl_rl.ppo_warehouse_jetbot_cfg:PPO_WAREHOUSE_JETBOT_RUNNER_CFG",
    },
)

# # navisim_isaacsim/__init__.py

# """IsaacLab integration for NaviSim navigation environments."""

# import os
# import gymnasium as gym

# # Import environment and config classes
# from navisim_lab.envs.warehouse.warehouse_env import WarehouseEnv
# from navisim_lab.envs.warehouse.warehouse_env_cfg import WarehouseEnvCfg
# from navisim_lab.configs.rsl_rl.ppo_cfg import PPO_WAREHOUSE_JETBOT_CFG

# # Register environment with Gymnasium including RL agent configurations
# gym.register(
#     id="Isaac-NavisimNavigation-Jetbot-v0",
#     entry_point="navisim_isaacsim.tasks.navigation_env:WarehouseEnv",
#     disable_env_checker=True,  # IsaacLab handles its own validation
#     kwargs={
#         "env_cfg_entry_point": WarehouseEnvCfg,
#         "rl_cfg_entry_point": PPO_WAREHOUSE_JETBOT_CFG, 
#     },
# )

# __all__ = [
#     "WarehouseEnv",
#     "WarehouseEnvCfg",
#     "PPO_WAREHOUSE_JETBOT_CFG",
# ]
