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