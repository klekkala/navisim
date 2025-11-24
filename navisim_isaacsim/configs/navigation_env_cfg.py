# configs/navigation_env_cfg.py

"""Configuration for Navisim navigation environment."""

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass
from scene.warehouse_scene_cfg import NavisimWarehouseSceneCfg
import isaaclab.sim as sim_utils


@configclass
class NavisimNavigationEnvCfg(DirectRLEnvCfg):
    """Configuration for Navisim navigation environment with Jetbot."""

    # Simulation settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,  # 60 Hz simulation
        render_interval=1,  # Render every step (required for camera capture)
        device="cuda:0",  # Isaac Sim requires CUDA
    )

    # Scene configuration
    scene: NavisimWarehouseSceneCfg = NavisimWarehouseSceneCfg(
        num_envs=1,
        env_spacing=4.0,
    )

    # Environment settings (inherited from DirectRLEnvCfg)
    decimation: int = 2  # Number of sim steps per env step
    episode_length_s: float = 10.0  # Episode length in seconds

    # Observation and action space dimensions
    # Observation: Jetbot root state [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13D
    observation_space: int = 13
    # Action: Wheel velocities [left_wheel, right_wheel] = 2D
    action_space: int = 2
    # State space (for asymmetric actor-critic, 0 = not used)
    state_space: int = 0

    # Task-specific settings
    action_scale: float = 50.0  # Scale for wheel velocities (rad/s) - higher for scaled robot

    # Reward weights
    forward_reward_weight: float = 1.0

    # Camera view (custom settings)
    viewer_eye: tuple[float, float, float] = (5.0, 0.0, 4.0)
    viewer_target: tuple[float, float, float] = (0.0, 0.0, 1.0)
