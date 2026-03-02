import gymnasium as gym

# Register Gym environment ID -> Isaac Lab environment class
# Isaac Lab recommends using gym.register with env_cfg_entry_point in kwargs.
gym.register(
    id="Navisim-Warehouse-Jetbot-v0",
    entry_point="navisim_lab.envs.warehouse.warehouse_env:WarehouseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "navisim_lab.envs.warehouse.warehouse_env_cfg:WarehouseEnvCfg",
        # optional:
        "rsl_rl_cfg_entry_point": "navisim_lab.configs.rsl_rl.ppo_cfg:PPO_WAREHOUSE_JETBOT_CFG",
    },
)

# Outdoor (3DGS) — no camera. Safe for headless RL training without --enable_cameras.
gym.register(
    id="Navisim-Outdoor-Jetbot-v0",
    entry_point="navisim_lab.envs.warehouse.warehouse_env:WarehouseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "navisim_lab.envs.outdoor.outdoor_env_cfg:OutdoorEnvCfg",
        "rsl_rl_cfg_entry_point": "navisim_lab.configs.rsl_rl.ppo_cfg:PPO_WAREHOUSE_JETBOT_CFG",
    },
)

# Outdoor (3DGS) — WITH Jetbot POV camera. Requires --enable_cameras.
gym.register(
    id="Navisim-Outdoor-Jetbot-Camera-v0",
    entry_point="navisim_lab.envs.warehouse.warehouse_env:WarehouseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "navisim_lab.envs.outdoor.outdoor_env_cfg:OutdoorEnvWithCameraCfg",
        "rsl_rl_cfg_entry_point": "navisim_lab.configs.rsl_rl.ppo_cfg:PPO_WAREHOUSE_JETBOT_CFG",
    },
)