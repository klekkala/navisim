"""Configuration for the custom outdoor navigation environment.

Two variants:
  OutdoorEnvCfg             — no camera (RL training / evaluation, headless-safe)
  OutdoorEnvWithCameraCfg   — adds POV camera (data collection, --enable_cameras required)
"""

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from navisim_lab.envs.outdoor.outdoor_scene_cfg import OutdoorSceneCfg


@configclass
class OutdoorEnvCfg(DirectRLEnvCfg):
    """Outdoor env without camera — safe to run headless without --enable_cameras."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=2,  # equals decimation — one render call per env step
        device="cuda:0",
    )

    scene: OutdoorSceneCfg = OutdoorSceneCfg(
        num_envs=1,
        env_spacing=4.0,
    )

    decimation: int = 2
    episode_length_s: float = 10.0

    observation_space: int = 13
    action_space: int = 2
    state_space: int = 0

    action_scale: float = 5.0
    forward_reward_weight: float = 1.0

    viewer_eye: tuple[float, float, float] = (0.0, -8.0, 6.0)
    viewer_target: tuple[float, float, float] = (0.0, 1.0, 0.0)


# OutdoorEnvWithCameraCfg lives in outdoor_env_with_camera_cfg.py for the same
# reason OutdoorSceneWithCameraCfg was moved — to avoid triggering the camera
# import (and rtx_sensors=True) when only the no-camera env is used.
