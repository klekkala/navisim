# Import Camera configuration
from isaaclab.sensors.camera import CameraCfg

jetbot_pov_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Jetbot/chassis/rgb_camera/jetbot_camera",
        spawn=None,  # Don't spawn - reference existing camera in USD
        update_latest_camera_pose=True,  # CRITICAL: Track camera transforms with XFormPrimView
        data_types=["rgb"],
        width=640,
        height=480,
    )