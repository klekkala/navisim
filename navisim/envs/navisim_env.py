import uuid
import gymnasium as gym
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Optional

try:
    from ..bridge.client import NavisimBridgeClient
    from ..config.gaussian_model_param import GaussianModelParam
    from ..data.rocksdb import reset_db
    from ..envs.game_window import GameWindow
    from ..motion.simple_motion_model import SimpleMotionModel
    from ..render.navisim_camera import NavisimCamera
    from ..render.navisim_scene import NavisimScene
    from ..structs.enums.enums import RenderMode
    from ..world.sequence_graph import SequenceGraph
except ImportError:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from navisim.config.gaussian_model_param import GaussianModelParam
    from navisim.data.rocksdb import reset_db
    from navisim.envs.game_window import GameWindow
    from navisim.motion.simple_motion_model import SimpleMotionModel
    from navisim.render.navisim_camera import NavisimCamera
    from navisim.render.navisim_scene import NavisimScene
    from navisim.structs.enums.enums import RenderMode
    from navisim.world.sequence_graph import SequenceGraph
    
# --------------------
# Helper structures
# --------------------
@dataclass
class AgentState:
    pose_world_T_agent: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    linear: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    angular: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

@dataclass
class FrameCache:
    rgb: Optional[np.ndarray] = None
    inv_depth: Optional[np.ndarray] = None
    render_ms: float = 0.0

# --------------------
# Environment
# --------------------
class NavisimEnv(gym.Env):
    """
    Production-clean Gymnasium env that:
    - Uses the Omniverse NavisimBridge for physics + rendering per step
    - Keeps Gaussian splat content management on the same bridge (control RPCs)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        sequence_graph: SequenceGraph,
        render_mode: RenderMode,
        window: Optional[GameWindow] = None,
        dt: float = 1.0 / 60.0,
    ):
        super().__init__()
        self.sequence_graph = sequence_graph
        self.render_mode = render_mode.name.lower()
        self.window = window
        self.dt = float(dt)

        # Bridge (single endpoint for physics + rendering + control)
        self.bridge = NavisimBridgeClient()

        # Scene & camera initialized in reset()
        self.camera: Optional[NavisimCamera] = None
        self.scene: Optional[NavisimScene] = None

        # Sequence/sector bookkeeping
        self.current_sequence_id: Optional[str] = None
        self.sequence_index: int = 0
        self.current_sector = None

        # Agent state
        self.state = AgentState()
        self.frame = FrameCache()
        self.current_step = 0
        self._target_locations: np.ndarray = np.zeros((0, 2), dtype=np.float32)

        # --------------------------
        # Gym spaces (tunable)
        # --------------------------
        # Actions: Discrete(3) → mapped to simple {stop, forward, turn-left}
        self.action_space = gym.spaces.Discrete(4)

        # Observations: RGB frame (uint8 HxWx3), pose (7 or 16?), linear, angular
        # Keep it simple for now: return a dict; define spaces for training-time validation
        H = getattr(self.window, "height", 720)
        W = getattr(self.window, "width", 1280)
        self.observation_space = gym.spaces.Dict({
            "rgb": gym.spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8),
            "pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float64),
            "vel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "omega": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })


    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        reset_db()

        # Pick a sequence and sector deterministically if seed provided
        rng = random.Random(seed)
        self.current_sequence_id = rng.choice(self.sequence_graph.get_sequence_ids())
        sequence = self.sequence_graph.get_sequence(self.current_sequence_id)

        # Unload previous
        if self.current_sector:
            try:
                self.current_sector.unload_all()
            except Exception:
                pass

        self.current_sector = sequence[self.sequence_index]

        # Camera/scene setup (scene mostly carries intrinsics & metadata now)
        W = getattr(self.window, "width", 1280)
        H = getattr(self.window, "height", 720)
        self.camera = NavisimCamera.create(
            camera_id=str(uuid.uuid4()), W=W, H=H
        )
        self.scene = NavisimScene.create(
            model_params=GaussianModelParam.create(model_path=self.current_sector.gaussian_model.model_path),
            camera=self.camera,
            sector=self.current_sector,
        )

        # Agent spawn
        self.state = AgentState(pose_world_T_agent=self.scene.random_start_location())
        self.current_step = 0

        # Targets (Nx2)
        self._target_locations = np.asarray(self.scene.get_target_locations(), dtype=np.float32)

        # Preload/activate sector on the renderer/bridge
        self._send_sector_activation(self.current_sector)

        # First observation: do one no-op step_render to get a frame aligned with physics
        obs = self._step_render_with_action(self._discrete_to_action(0))  # 0 → no-op/stop
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        obs = self._step_render_with_action(self._discrete_to_action(action))
        reward, terminated = self._compute_reward_and_done()
        truncated = False
        info = self._get_info()
        self.current_step += 1
        return obs, reward, terminated, truncated, info

    def render(self):
        """Return last RGB (fast) or show in window for human mode."""
        if self.render_mode == "human" and self.window and self.frame.rgb is not None:
            # If you want an on-screen preview, hand your RGB to the window here.
            # self.window.display_images(self.frame.rgb, None)
            pass
        return self.frame.rgb

    def close(self):
        pass

    # -------------
    # Internals
    # -------------
    def _camera_dict(self) -> Dict[str, Any]:
        return {
            "width": self.camera.W,
            "height": self.camera.H,
            "fx": self.camera.fx,
            "fy": self.camera.fy,
            "cx": self.camera.cx,
            "cy": self.camera.cy,
        }

    def _current_sector_id_version(self) -> Tuple[str, str]:
        sid = f"{self.current_sector.seq_id}/{self.current_sector.sector_id}"
        ver = getattr(self.current_sector, "version", "v0")
        return sid, ver

    def _step_render_with_action(self, action_msg: Dict[str, Any]) -> Dict[str, np.ndarray]:
        cam = self._camera_dict()
        sid, ver = self._current_sector_id_version()

        header, color, inv_depth = self.bridge.step_render(
            action=action_msg, dt=self.dt, camera=cam, sector_id=sid, version=ver
        )

        # Authoritative physics state → agent state
        st = header["state"]
        self.state.pose_world_T_agent = np.asarray(st["T_world_agent"], dtype=np.float64)
        self.state.linear = np.asarray(st["linear"], dtype=np.float32)
        self.state.angular = np.asarray(st["angular"], dtype=np.float32)

        # Cache frame
        self.frame.rgb = color
        self.frame.inv_depth = inv_depth
        self.frame.render_ms = float(header.get("render_ms", 0.0))

        return {
            "rgb": self.frame.rgb,
            "pose": self.state.pose_world_T_agent,
            "vel": self.state.linear,
            "omega": self.state.angular,
        }

    def _compute_reward_and_done(self) -> Tuple[float, bool]:
        """
        Example shaping: small negative step cost; terminal reward when close to a target.
        Replace with your task’s real logic.
        """
        pos_xy = self._pose_xy(self.state.pose_world_T_agent)
        if self._target_locations.size:
            # L1 distance to closest target
            dists = np.abs(self._target_locations - pos_xy).sum(axis=1)
            closest = float(np.min(dists))
        else:
            closest = 1e9

        terminated = closest < 0.5  # within 0.5 m (tune)
        reward = (1.0 if terminated else 0.0) - 0.01  # step penalty
        return reward, terminated

    def _get_info(self) -> Dict[str, Any]:
        pos_xy = self._pose_xy(self.state.pose_world_T_agent)
        if self._target_locations.size:
            dists = np.abs(self._target_locations - pos_xy).sum(axis=1)
            closest = float(np.min(dists))
        else:
            closest = float("inf")
        return {
            "distance": closest,
            "num_targets": int(self._target_locations.shape[0]),
            "agent_pose": self.state.pose_world_T_agent.copy(),
            "render_ms": self.frame.render_ms,
        }

    def _send_sector_activation(self, sector) -> None:
        """Ask the bridge/renderer to preload + activate the sector (best-effort)."""
        sid = f"{sector.seq_id}/{sector.sector_id}"
        ver = getattr(sector, "version", "v0")
        try:
            self.bridge.control("load_sector", sector_id=sid, version=ver)
            self.bridge.control("activate_sector", sector_id=sid, version=ver)
        except Exception as e:
            # Non-fatal; renderer may already have it
            print(f"[bridge] sector activation warning: {e}")

    # ----------------
    # Utilities
    # ----------------
    @staticmethod
    def _pose_xy(T_world_agent: np.ndarray) -> np.ndarray:
        """Extract (x,y) translation from a 4x4 world_T_agent."""
        return np.array([T_world_agent[0, 3], T_world_agent[1, 3]], dtype=np.float32)

    def _discrete_to_action(self, a: int) -> Dict[str, Any]:
        """
        Map Discrete(4) → bridge action message.
        0: stop, 1: forward, 2: left, 3: right
        """
        if a == 1:      # forward
            return {"mode": "vel", "linear": [1.0, 0.0, 0.0], "angular": [0.0, 0.0, 0.0]}
        elif a == 2:    # turn left
            return {"mode": "vel", "linear": [0.0, 0.0, 0.0], "angular": [0.0, 0.0, +0.5]}
        elif a == 3:    # turn right
            return {"mode": "vel", "linear": [0.0, 0.0, 0.0], "angular": [0.0, 0.0, -0.5]}
        else:           # stop
            return {"mode": "vel", "linear": [0.0, 0.0, 0.0], "angular": [0.0, 0.0, 0.0]}