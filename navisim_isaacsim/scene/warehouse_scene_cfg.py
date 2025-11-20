# scene/warehouse_scene_cfg.py
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.sensors import PinholeCameraCfg

from assets.jetbot_cfg import JETBOT_CONFIG

@configclass
class NavisimWarehouseSceneCfg(InteractiveSceneCfg):

    jetbot = JETBOT_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Jetbot",
    )

    front_cam = CameraCfg(
            prim_path="/World/envs/env_.*/Jetbot/camera",
            width=256,
            height=256,   
            spawn=PinholeCameraCfg(     # intrinsics
                focal_length=2.0,
                horizontal_aperture=2.4,
                vertical_aperture=3.6,
                clipping_range=(0.01, 100.0),
            ),
    )
