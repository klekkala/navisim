# scene/warehouse_scene_with_camera_cfg.py

"""Scene configuration with Jetbot robot and camera sensor."""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, patterns
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from assets.jetbot_cfg import JETBOT_CONFIG


@configclass
class NavisimWarehouseSceneWithCameraCfg(InteractiveSceneCfg):
    """Scene configuration with Jetbot robot and camera sensor.

    This configuration adds an Isaac Lab Camera sensor that wraps the
    Jetbot's built-in front_cam USD prim, making it accessible for
    observations and image capture.
    """

    jetbot = JETBOT_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Jetbot",
    )

    # Camera sensor wrapping Jetbot's built-in camera
    front_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Jetbot/chassis/front_cam",
        update_period=0.1,  # Update at 10 Hz (every 0.1 seconds)
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],  # RGB image and depth
        spawn=None,  # Don't spawn - camera already exists in Jetbot USD
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),  # No offset - use existing position
            rot=(1.0, 0.0, 0.0, 0.0),  # No rotation
            convention="world",
        ),
    )
