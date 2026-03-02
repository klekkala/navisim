"""Outdoor env config WITH Jetbot POV camera.

Kept in a separate file from outdoor_env_cfg.py so that importing the
no-camera env config never triggers the isaaclab.sensors.camera import.

Only import this module when launching Navisim-Outdoor-Jetbot
with --enable_cameras.
"""

from isaaclab.utils import configclass

from navisim_lab.envs.outdoor.outdoor_env_cfg import OutdoorEnvCfg
from navisim_lab.envs.outdoor.outdoor_scene_with_camera_cfg import OutdoorSceneWithCameraCfg


@configclass
class OutdoorEnvWithCameraCfg(OutdoorEnvCfg):
    """Outdoor env WITH Jetbot POV camera.

    Use only with --enable_cameras (AppLauncher).
    render_interval=2 = decimation ensures exactly one sim.render() per env step,
    preventing the Replicator annotator from blocking on a missing frame.
    """

    scene: OutdoorSceneWithCameraCfg = OutdoorSceneWithCameraCfg(
        num_envs=1,
        env_spacing=4.0,
    )
