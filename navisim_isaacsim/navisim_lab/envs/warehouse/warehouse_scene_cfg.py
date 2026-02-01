# scene/warehouse_scene_cfg.py
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

from navisim_lab.robots.jetbot_cfg import JETBOT_CONFIG
from navisim_lab.utils.paths import WAREHOUSE_USD
from navisim_lab.camera.jetbot_camera import jetbot_pov_camera

@configclass
class WarehouseSceneCfg(InteractiveSceneCfg):
    """Scene configuration with Jetbot robot and warehouse environment.

    The Jetbot USD includes a built-in camera at chassis/rgb_camera/jetbot_camera.
    We reference it directly using spawn=None and update_latest_camera_pose=True
    to ensure the camera properly tracks the Jetbot's movement.
    """

    # Warehouse environment (spawned once globally at /World/Warehouse)
    warehouse = AssetBaseCfg(
        prim_path="/World/Warehouse",
        spawn=sim_utils.UsdFileCfg(usd_path=WAREHOUSE_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Note: Ground plane and dome light are already created by Isaac Lab's DirectRLEnv
    # We don't need to add them here to avoid conflicts

    # Jetbot robot (spawned per environment using ENV_REGEX_NS)
    jetbot = JETBOT_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Jetbot",
    )

    # Jetbot onboard camera (reference existing camera prim in Jetbot USD)
    # The Jetbot USD file includes a camera at chassis/rgb_camera/jetbot_camera
    # We use spawn=None to reference it (not spawn it) and update_latest_camera_pose=True
    # to ensure XFormPrimView tracks the camera's position as Jetbot moves
    jetbot_camera = jetbot_pov_camera
