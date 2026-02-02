"""Base navigation environment for NaviSim tasks."""

from abc import abstractmethod
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene


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
        # Previous position for computing displacement
        self.prev_position = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float32
        )

        # Goal positions (if applicable to the task)
        self.goal_positions = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float32
        )

        # Flags for task-specific events
        self.collision_flags = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def _setup_scene(self):
        """Setup the scene with standard configuration.

        This method:
        1. Creates InteractiveScene from config
        2. Clones environments for parallelization
        3. Resets simulation
        4. Sets camera view

        Override if you need custom scene setup behavior.
        """
        # Create scene from config
        self.scene = InteractiveScene(self.cfg.scene)

        # Clone environments (create num_envs copies)
        self.scene.clone_environments(copy_from_source=False)

        # Add scene entities to simulation
        self.sim.reset()

        # Set camera view from config
        self.sim.set_camera_view(eye=self.cfg.viewer_eye, target=self.cfg.viewer_target)

    @abstractmethod
    def _compute_navigation_reward(self) -> torch.Tensor:
        """Compute task-specific navigation reward.

        Returns:
            Reward tensor for each environment. Shape: (num_envs,)

        Example:
            # Forward progress reward
            curr_pos = self.robot.data.root_state_w[:, :3]
            displacement = curr_pos - self.prev_position
            reward = displacement[:, 0] * self.cfg.forward_weight
            self.prev_position = curr_pos.clone()
            return reward
        """
        pass

    @abstractmethod
    def _check_termination_conditions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check for task-specific early termination conditions.

        Returns:
            Tuple of (terminated, truncated) boolean tensors:
            - terminated: Episode ended due to task completion/failure
            - truncated: Episode ended due to time limit (computed automatically)

        Example:
            # Check for collision or goal reached
            terminated = self.collision_flags | self.goal_reached_flags
            truncated = torch.zeros_like(terminated)
            return terminated, truncated
        """
        pass

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards by delegating to task-specific implementation.

        Returns:
            Reward tensor. Shape: (num_envs,)
        """
        return self._compute_navigation_reward()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine episode termination by combining timeout and task conditions.

        Returns:
            Tuple of (terminated, truncated) boolean tensors
        """
        # Calculate max episode steps
        max_episode_steps = int(
            self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation)
        )

        # Check timeout
        time_out = self.episode_length_buf >= max_episode_steps - 1

        # Check task-specific termination conditions
        terminated, task_truncated = self._check_termination_conditions()

        # Combine truncation conditions
        truncated = time_out | task_truncated

        return terminated, truncated

    def _compute_distance_to_goal(self, robot_positions: torch.Tensor) -> torch.Tensor:
        """Compute distance from robot to goal positions.

        Args:
            robot_positions: Robot positions tensor. Shape: (num_envs, 3)

        Returns:
            Distance tensor. Shape: (num_envs,)
        """
        return torch.norm(robot_positions - self.goal_positions, dim=1)

    def _compute_forward_progress(
        self, current_pos: torch.Tensor, axis: int = 0
    ) -> torch.Tensor:
        """Compute forward progress along a specified axis.

        Args:
            current_pos: Current positions. Shape: (num_envs, 3)
            axis: Axis index (0=x, 1=y, 2=z)

        Returns:
            Progress tensor. Shape: (num_envs,)
        """
        progress = current_pos[:, axis] - self.prev_position[:, axis]
        self.prev_position = current_pos.clone()
        return progress

    def _reset_navigation_state(self, env_ids: torch.Tensor):
        """Reset common navigation state for specified environments.

        Call this from _reset_idx() in subclasses.

        Args:
            env_ids: Indices of environments to reset
        """
        # Reset position tracking
        # Note: Subclasses should update prev_position with actual robot position

        # Reset collision flags
        self.collision_flags[env_ids] = False

        # Reset goals (if used by the task)
        # Note: Subclasses should set goal_positions if applicable

    def _set_debug_vis(self, debug_vis: bool) -> None:
        """Enable/disable debug visualization.

        Override to add custom debug visualizations (goal markers, trajectories, etc.)

        Args:
            debug_vis: Whether to enable debug visualization
        """
        # Base implementation does nothing
        # Subclasses can add custom visualizations here
        pass
