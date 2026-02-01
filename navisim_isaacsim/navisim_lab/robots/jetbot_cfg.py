import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from navisim_lab.utils.paths import JETBOT_USD

JETBOT_CONFIG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Jetbot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=JETBOT_USD,
        scale=(7.0, 7.0, 7.0),  # Scale up to human size (like in working example)
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.21),  # Spawn with wheels on ground (wheel radius = 0.03 * 7 = 0.21m)
    ),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],  # Specific wheel joints
            stiffness=0.0,
            damping=1000.0,  # High damping to generate sufficient torque for velocity control on scaled robot
            velocity_limit=200.0,  # Higher velocity for scaled robot
            effort_limit=5000.0,   # Much higher effort for scaled 7x robot (mass scales with volume = 7^3)
        )
    },
)
