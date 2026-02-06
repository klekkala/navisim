"""Base navigation environment for NaviSim tasks."""

from abc import abstractmethod
import logging
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene

logger = logging.getLogger(__name__)


class BaseNavigationEnv(DirectRLEnv):
    """Base class for navigation environments in NaviSim.

    This class provides common functionality for navigation tasks, including:
    - Standard scene setup with camera positioning
    - Common navigation state management (position tracking, goal handling)
    - Shared termination logic (timeout, collision detection hooks)
    - Utility methods for reward computation

    Subclasses should implement:
    - _compute_navigation_reward(): Task-specific reward logic
    - _check_termination_conditions(): Early termination conditions
    - _pre_physics_step(): Action processing
    - _get_observations(): Observation computation
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        """Initialize base navigation environment.

        Args:
            cfg: Environment configuration (must inherit from DirectRLEnvCfg)
            render_mode: Rendering mode for the environment
            **kwargs: Additional arguments passed to DirectRLEnv
        """
        super().__init__(cfg, render_mode, **kwargs)

        # Common navigation state tensors
        self._initialize_task_state()

    def _initialize_task_state(self):
        """Initialize common task-specific state tensors.

        Override this method to add custom state tracking.
        Remember to call super()._initialize_task_state() in subclasses.
        """
        self.prev_position = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float32
        )
        self.goal_positions = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float32
        )
        self.collision_flags = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def _setup_scene(self):
        """Setup the scene with standard configuration.

        Order of operations:
        1. Create InteractiveScene from config
        2. Clone environments for parallelization
        3. Standardize camera transforms (if camera is in scene)
        4. Reset simulation (sensors initialize here)
        5. Set viewer camera
        """
        self.scene = InteractiveScene(self.cfg.scene)
        self.scene.clone_environments(copy_from_source=False)

        # Fix camera transforms BEFORE sim.reset() - sensors initialize during reset
        self._standardize_camera_transforms()

        self.sim.reset()
        self.sim.set_camera_view(eye=self.cfg.viewer_eye, target=self.cfg.viewer_target)

    def _standardize_camera_transforms(self):
        """Standardize camera prim transforms from rotateZYX to orient (quaternion).

        Isaac Lab requires camera prims to have canonical transform ops
        [translate, orient, scale]. The Jetbot USD uses rotateZYX which causes
        XformPrimView validation errors during camera sensor initialization.

        This method uses the Sdf layer API to override composed USD properties
        from references, since the Usd API cannot modify properties authored
        in referenced layers (e.g., the Jetbot USD file).

        Must run AFTER clone_environments() and BEFORE sim.reset().
        """
        if not hasattr(self.cfg.scene, 'jetbot_camera'):
            return

        from pxr import UsdGeom, Gf, Sdf

        stage = self.sim.stage
        root_layer = stage.GetRootLayer()

        for env_id in range(self.num_envs):
            prim_path = f"/World/envs/env_{env_id}/Jetbot/chassis/rgb_camera/jetbot_camera"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                if env_id == 0:
                    logger.warning(f"Camera prim not found at {prim_path}")
                continue

            xformable = UsdGeom.Xformable(prim)

            # Read current local transform (captures rotateZYX as a matrix)
            tf = Gf.Transform(xformable.GetLocalTransformation())
            pos = Gf.Vec3d(tf.GetTranslation())
            quat = Gf.Quatd(tf.GetRotation().GetQuat())
            scale = Gf.Vec3d(tf.GetScale())

            # Override transform ops at the Sdf layer level
            sdf_path = Sdf.Path(prim_path)
            sdf_prim = root_layer.GetPrimAtPath(sdf_path)

            if sdf_prim is None:
                sdf_prim = Sdf.CreatePrimInLayer(root_layer, sdf_path)
                sdf_prim.specifier = Sdf.SpecifierOver

            # Remove rotateZYX if authored in this layer
            if "xformOp:rotateZYX" in sdf_prim.properties:
                sdf_prim.RemoveProperty(sdf_prim.properties["xformOp:rotateZYX"])

            # Override xformOpOrder to canonical form
            xform_op_order_attr = sdf_prim.attributes.get("xformOpOrder")
            if xform_op_order_attr is None:
                xform_op_order_attr = Sdf.AttributeSpec(
                    sdf_prim, "xformOpOrder", Sdf.ValueTypeNames.TokenArray
                )
            xform_op_order_attr.default = [
                "xformOp:translate", "xformOp:orient", "xformOp:scale"
            ]

            # Add orient attribute (quaternion) if not present
            if "xformOp:orient" not in sdf_prim.properties:
                orient_attr = Sdf.AttributeSpec(
                    sdf_prim, "xformOp:orient", Sdf.ValueTypeNames.Quatd
                )
                orient_attr.default = quat

            # Ensure translate and scale exist
            if "xformOp:translate" not in sdf_prim.properties:
                translate_attr = Sdf.AttributeSpec(
                    sdf_prim, "xformOp:translate", Sdf.ValueTypeNames.Double3
                )
                translate_attr.default = pos

            if "xformOp:scale" not in sdf_prim.properties:
                scale_attr = Sdf.AttributeSpec(
                    sdf_prim, "xformOp:scale", Sdf.ValueTypeNames.Double3
                )
                scale_attr.default = scale

            if env_id == 0:
                new_ops = [op.GetOpName() for op in xformable.GetOrderedXformOps()]
                logger.info(f"Standardized camera transforms: {new_ops}")

    @abstractmethod
    def _compute_navigation_reward(self) -> torch.Tensor:
        """Compute task-specific navigation reward.

        Returns:
            Reward tensor for each environment. Shape: (num_envs,)
        """
        pass

    @abstractmethod
    def _check_termination_conditions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check for task-specific early termination conditions.

        Returns:
            Tuple of (terminated, truncated) boolean tensors.
        """
        pass

    def _get_rewards(self) -> torch.Tensor:
        return self._compute_navigation_reward()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_episode_steps = int(
            self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation)
        )
        time_out = self.episode_length_buf >= max_episode_steps - 1
        terminated, task_truncated = self._check_termination_conditions()
        truncated = time_out | task_truncated
        return terminated, truncated

    def _compute_distance_to_goal(self, robot_positions: torch.Tensor) -> torch.Tensor:
        """Compute distance from robot to goal positions."""
        return torch.norm(robot_positions - self.goal_positions, dim=1)

    def _compute_forward_progress(
        self, current_pos: torch.Tensor, axis: int = 0
    ) -> torch.Tensor:
        """Compute forward progress along a specified axis."""
        progress = current_pos[:, axis] - self.prev_position[:, axis]
        self.prev_position = current_pos.clone()
        return progress

    def _reset_navigation_state(self, env_ids: torch.Tensor):
        """Reset common navigation state for specified environments.

        Call this from _reset_idx() in subclasses.
        """
        self.collision_flags[env_ids] = False

    def _set_debug_vis(self, debug_vis: bool) -> None:
        pass
