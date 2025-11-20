# scene/warehouse_scene_cfg_simple.py
"""Simple warehouse scene without camera for initial testing."""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from assets.jetbot_cfg import JETBOT_CONFIG


@configclass
class NavisimWarehouseSceneCfg(InteractiveSceneCfg):
    """Simple scene with just Jetbot - no camera."""

    jetbot = JETBOT_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Jetbot",
    )
