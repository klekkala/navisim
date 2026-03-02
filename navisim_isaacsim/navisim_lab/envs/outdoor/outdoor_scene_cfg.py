"""Scene configurations for custom 3DGS-converted outdoor environment.

Two variants are provided:
  OutdoorSceneCfg             — no camera sensor (RL training, fast, no RTX overhead)
  OutdoorSceneWithCameraCfg   — adds jetbot_camera (camera capture / data collection)

WHY the split:
  Isaac Lab's Camera sensor sets the carb flag /isaaclab/render/rtx_sensors=True the
  moment it is instantiated. This causes sim.render() to be called every render_interval
  physics steps in the DirectRLEnv step loop. In headless mode without --enable_cameras
  the offscreen renderer is not initialised, so the Replicator annotator blocks forever
  waiting for a frame — i.e. the simulation hangs.

  By keeping the camera out of the default scene config, training and evaluation without
  camera capture run at full speed.  Only the camera-capture script uses
  OutdoorSceneWithCameraCfg + OutdoorEnvWithCameraCfg (render_interval=2 = decimation).

The USDZ is purely visual (no collision geometry, no rigid bodies) because it was
converted from a 3D Gaussian Splatting reconstruction. Physics is handled by:
  - SimulationContext (created by DirectRLEnv) provides the PhysicsScene automatically
  - An invisible ground plane provides the collision surface for the Jetbot
"""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

from navisim_lab.robots.jetbot_cfg import JETBOT_CONFIG
from navisim_lab.utils.paths import CUSTOM_ENV_USD, JETBOT_USD
from navisim_lab.camera.jetbot_camera import jetbot_pov_camera

# The outdoor scene is ~4.3m x 6.6m (meters per unit = 1.0).
# Jetbot base wheel radius = 0.03m. At scale=1.0:
#   - Body width  ~0.12m
#   - Spawn height = 0.03 * 1.0 = 0.03m
_OUTDOOR_JETBOT_SCALE = 1.0
_OUTDOOR_JETBOT_WHEEL_RADIUS = 0.03 * _OUTDOOR_JETBOT_SCALE

OUTDOOR_JETBOT_CONFIG = JETBOT_CONFIG.replace(
    spawn=sim_utils.UsdFileCfg(
        usd_path=JETBOT_USD,
        scale=(_OUTDOOR_JETBOT_SCALE,) * 3,
    ),
    init_state=JETBOT_CONFIG.InitialStateCfg(
        pos=(0.0, 0.0, _OUTDOOR_JETBOT_WHEEL_RADIUS),
    ),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            stiffness=0.0,
            damping=200.0,
            velocity_limit=200.0,
            effort_limit=500.0,
        )
    },
)


@configclass
class OutdoorSceneCfg(InteractiveSceneCfg):
    """Base outdoor scene — NO camera sensor.

    Use this for RL training and evaluation.  No RTX render products are created
    so the simulation does not hang in headless mode.

    Physics:
    - ground_plane: the only collision surface (USDZ has no physics geometry).
    - outdoor_env: USDZ loaded as a visual-only reference.
    - jetbot: differential-drive robot scaled to fit the ~4m x 6m scene.
    """

    ground_plane = AssetBaseCfg(
        prim_path="/World/OutdoorGround",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    outdoor_env = AssetBaseCfg(
        prim_path="/World/OutdoorEnv",
        spawn=sim_utils.UsdFileCfg(usd_path=CUSTOM_ENV_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    jetbot = OUTDOOR_JETBOT_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Jetbot",
    )


@configclass
class OutdoorSceneWithCameraCfg(OutdoorSceneCfg):
    """Outdoor scene WITH Jetbot POV camera sensor.

    Use this only when camera images are needed (data collection, visualisation).
    Requires --enable_cameras to be set in AppLauncher and render_interval=decimation
    in SimulationCfg to avoid annotator frame-wait hangs.

    Camera transforms are standardised (rotateZYX -> orient) by
    BaseNavigationEnv._standardize_camera_transforms() before sim.reset().
    """

    jetbot_camera = jetbot_pov_camera
