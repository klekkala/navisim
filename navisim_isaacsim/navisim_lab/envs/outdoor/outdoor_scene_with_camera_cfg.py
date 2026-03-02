"""Outdoor scene config WITH Jetbot POV camera.

Kept in a separate file from outdoor_scene_cfg.py so that importing the
no-camera variant never triggers the isaaclab.sensors.camera import.
That import sets the carb flag /isaaclab/render/rtx_sensors=True, which
activates the RTX offscreen renderer and causes headless runs (without
--enable_cameras) to hang indefinitely.

Only import this module when you actually need camera capture, i.e. when
using Navisim-Outdoor-Jetbot-Camera-v0 with --enable_cameras.
"""

from isaaclab.utils import configclass

from navisim_lab.camera.jetbot_camera import jetbot_pov_camera
from navisim_lab.envs.outdoor.outdoor_scene_cfg import OutdoorSceneCfg


@configclass
class OutdoorSceneWithCameraCfg(OutdoorSceneCfg):
    """Outdoor scene WITH Jetbot POV camera sensor.

    Use only with --enable_cameras (AppLauncher) and render_interval=decimation
    in SimulationCfg to avoid Replicator annotator frame-wait hangs.

    Camera transforms are standardised (rotateZYX -> orient) by
    BaseNavigationEnv._standardize_camera_transforms() before sim.reset().
    """

    jetbot_camera = jetbot_pov_camera
