# configs/navigation_env_cfg.py

"""Configuration for Navisim navigation environment."""

from isaaclab.utils import configclass
from scene.warehouse_scene_cfg import NavisimWarehouseSceneCfg
import isaaclab.sim as sim_utils


@configclass
class NavisimNavigationEnvCfg:
    """Configuration for Navisim navigation environment with Jetbot."""

    # Simulation settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,  # 60 Hz simulation
        render_interval=2,  # Render every 2 steps
        device="cuda:0",
    )

    # Scene configuration
    scene: NavisimWarehouseSceneCfg = NavisimWarehouseSceneCfg(
        num_envs=1,
        env_spacing=4.0,
    )

    # Environment settings
    decimation: int = 2  # Number of sim steps per env step
    episode_length_s: float = 10.0  # Episode length in seconds

    # Action scaling
    action_scale: float = 10.0  # Scale for wheel velocities (rad/s)

    # Reward weights
    forward_reward_weight: float = 1.0

    # Camera view
    viewer_eye: tuple[float, float, float] = (5.0, 0.0, 4.0)
    viewer_target: tuple[float, float, float] = (0.0, 0.0, 1.0)
