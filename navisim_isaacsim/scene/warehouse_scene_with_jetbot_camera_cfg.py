# scene/warehouse_scene_with_jetbot_camera_cfg.py

"""Scene configuration with Jetbot robot and first-person camera."""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from assets.jetbot_cfg import JETBOT_CONFIG


@configclass
class NavisimWarehouseSceneWithJetbotCameraCfg(InteractiveSceneCfg):
    """Scene configuration with Jetbot robot and first-person camera.

    Note: The camera is spawned at the environment origin. The environment
    code should update the camera pose to follow the Jetbot each step.
    """

    jetbot = JETBOT_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Jetbot",
    )

    # First-person camera for Jetbot navigation
    # Spawned at env origin, will be moved to track Jetbot in environment code
    jetbot_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/JetbotCamera",
        update_period=0.1,  # Update at 10 Hz
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),  # Start at origin
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )
