# scene/warehouse_scene_with_third_person_camera_cfg.py

"""Scene configuration with Jetbot robot and third-person camera."""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from assets.jetbot_cfg import JETBOT_CONFIG


@configclass
class NavisimWarehouseSceneWithThirdPersonCameraCfg(InteractiveSceneCfg):
    """Scene configuration with Jetbot robot and a third-person camera.

    This adds a new camera above and behind the robot to capture navigation footage.
    Unlike the front_cam approach, this camera is spawned fresh and follows the robot.
    """

    jetbot = JETBOT_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Jetbot",
    )

    # Third-person camera that spawns above and behind the robot
    third_person_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ThirdPersonCamera",
        update_period=0.1,  # Update at 10 Hz
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1e6),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(2.0, 0.0, 1.5),  # 2m behind, 1.5m above robot
            rot=(0.9239, 0.0, 0.3827, 0.0),  # Angled down to look at robot
            convention="world",
        ),
    )
