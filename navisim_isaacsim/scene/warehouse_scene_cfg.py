# scene/warehouse_scene_cfg.py
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

from assets.jetbot_cfg import JETBOT_CONFIG
from configs.paths import WAREHOUSE_USD

@configclass
class NavisimWarehouseSceneCfg(InteractiveSceneCfg):
    """Scene configuration with Jetbot robot and warehouse environment.

    Note: Jetbot USD already includes a camera at chassis/front_cam.
    You can access it via USD prims if needed, but we don't declare it
    in the scene config since it's already part of the Jetbot asset.
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
