# scene/warehouse_scene_cfg.py
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

from navisim_lab.robots.jetbot_cfg import JETBOT_CONFIG
from navisim_lab.utils.paths import WAREHOUSE_USD

@configclass
class WarehouseSceneCfg(InteractiveSceneCfg):
    """Scene configuration with Jetbot robot and warehouse environment.

    Camera is intentionally omitted. Adding a CameraCfg activates the RTX
    offscreen renderer which causes the simulation to hang in headless mode
    unless --enable_cameras is passed to AppLauncher.
    """

    warehouse = AssetBaseCfg(
        prim_path="/World/Warehouse",
        spawn=sim_utils.UsdFileCfg(usd_path=WAREHOUSE_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    jetbot = JETBOT_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Jetbot",
    )
