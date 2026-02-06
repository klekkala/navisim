"""Warehouse scene configuration WITH camera support.

This is an alternative scene config that properly handles the Jetbot camera
by standardizing its transforms during scene setup.

To use this instead of the default warehouse_scene_cfg.py:
1. Import this in your environment config
2. Or copy the camera setup logic to your existing scene
"""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

from navisim_lab.robots.jetbot_cfg import JETBOT_CONFIG
from navisim_lab.utils.paths import WAREHOUSE_USD
from navisim_lab.camera.jetbot_camera import jetbot_pov_camera


@configclass
class WarehouseSceneWithCameraCfg(InteractiveSceneCfg):
    """Scene configuration with Jetbot robot, warehouse, and working camera.

    This version properly handles the camera transform issue by:
    1. Standardizing camera transforms during scene setup
    2. Using update_latest_camera_pose=True for accurate tracking
    """

    # Warehouse environment
    warehouse = AssetBaseCfg(
        prim_path="/World/Warehouse",
        spawn=sim_utils.UsdFileCfg(usd_path=WAREHOUSE_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Jetbot robot
    jetbot = JETBOT_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Jetbot",
    )

    # Jetbot camera with proper transform handling
    jetbot_camera = jetbot_pov_camera.replace(
        update_latest_camera_pose=True,  # Enable camera tracking
    )


# Helper function to standardize camera transforms after scene creation
def standardize_camera_transforms(env):
    """Standardize Jetbot camera transforms to work with Isaac Lab.

    Call this in your environment's _setup_scene() method after super()._setup_scene()

    Args:
        env: The environment instance with self.scene and self.sim

    Example:
        def _setup_scene(self):
            super()._setup_scene()
            standardize_camera_transforms(self)
    """
    import isaaclab.sim as sim_utils

    # Standardize transforms for all camera prims
    for env_id in range(env.num_envs):
        camera_prim_path = f"/World/envs/env_{env_id}/Jetbot/chassis/rgb_camera/jetbot_camera"

        try:
            # This converts rotateZYX to orient (quaternion)
            sim_utils.standardize_xform_ops(camera_prim_path, env.sim.stage)
            print(f"Standardized camera transforms for env {env_id}")
        except Exception as e:
            print(f"Warning: Could not standardize camera transforms for env {env_id}: {e}")
