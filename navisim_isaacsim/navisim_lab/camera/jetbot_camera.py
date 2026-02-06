"""Jetbot POV camera configuration for Isaac Lab.

The Jetbot USD uses rotateZYX transform operations on its camera prim, which is
incompatible with Isaac Lab's XformPrimView (requires [translate, orient, scale]).
BaseNavigationEnv._standardize_camera_transforms() fixes this at the Sdf layer
level before sim.reset().
"""
from isaaclab.sensors.camera import CameraCfg

jetbot_pov_camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Jetbot/chassis/rgb_camera/jetbot_camera",
    spawn=None,  # Reference existing camera prim in Jetbot USD
    update_latest_camera_pose=False,
    data_types=["rgb"],
    width=640,
    height=480,
)
