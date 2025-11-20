# scene/warehouse_scene_cfg.py
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from assets.jetbot_cfg import JETBOT_CONFIG

@configclass
class NavisimWarehouseSceneCfg(InteractiveSceneCfg):
    """Scene configuration with Jetbot robot.

    Note: Jetbot USD already includes a camera at chassis/front_cam.
    You can access it via USD prims if needed, but we don't declare it
    in the scene config since it's already part of the Jetbot asset.
    """

    jetbot = JETBOT_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Jetbot",
    )
