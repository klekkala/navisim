# tasks/navigation_env.py

"""IsaacLab RL environment for Jetbot navigation in warehouse."""

import torch
from navisim_lab.envs.base import BaseNavigationEnv
from navisim_lab.envs.warehouse.warehouse_env_cfg import WarehouseEnvCfg


class WarehouseEnv(BaseNavigationEnv):
    """
    IsaacLab RL environment for Jetbot navigation in warehouse.

    **Observations**: Jetbot root state (13D: position, orientation, velocities)

    **Actions**: Wheel velocities (2D: left_wheel, right_wheel) in range [-1, 1]

    **Rewards**: Forward progress along x-axis
    """

    cfg: WarehouseEnvCfg

    def __init__(self, cfg: WarehouseEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the navigation environment.

        Args:
            cfg: Configuration for the environment.
            render_mode: Render mode for the environment. Defaults to None.
            **kwargs: Additional keyword arguments passed to BaseNavigationEnv.
        """
        super().__init__(cfg, render_mode, **kwargs)

        # Note: Action and observation spaces are automatically created by DirectRLEnv
        # from cfg.action_space and cfg.observation_space dimensions

    def _initialize_task_state(self):
        """Initialize warehouse-specific state."""
        super()._initialize_task_state()

        # Task-specific state: track x-position for forward progress reward
        self.prev_x = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply actions before physics step.

        Args:
            actions: Normalized wheel velocities [-1, 1] for [left, right] wheels.
                    Shape: (num_envs, 2)
        """
        # Scale actions to actual wheel velocity range
        scaled_actions = actions * self.cfg.action_scale

        # Apply to Jetbot wheels
        self.scene["jetbot"].set_joint_velocity_target(scaled_actions)

    def _apply_action(self) -> None:
        """Write actions to simulation (called automatically by DirectRLEnv)."""
        self.scene.write_data_to_sim()

    def _get_observations(self) -> dict:
        """Compute observations for all environments.

        Returns:
            Dictionary with 'policy' key containing observations tensor.
            Observations are the Jetbot root state: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            Shape: (num_envs, 13)
        """
        # Get Jetbot root state
        obs = self.scene["jetbot"].data.root_state_w.clone()

        # DirectRLEnv expects dict with 'policy' key
        return {"policy": obs}

    def _compute_navigation_reward(self) -> torch.Tensor:
        """Compute rewards based on forward progress along x-axis.

        A teleport from a sector transition causes a discontinuous jump in X.
        Any step where |delta_x| > 1 m is treated as a teleport and returns 0
        reward; prev_x is also updated so the *following* step is clean.

        Returns:
            Reward tensor for each environment. Shape: (num_envs,)
        """
        curr_x = self.scene["jetbot"].data.root_state_w[:, 0]
        delta_x = curr_x - self.prev_x
        self.prev_x = curr_x.clone()

        # Zero reward on teleport steps (|delta| > 1 m is not physically reachable
        # in a single sim step at normal speeds)
        teleport_mask = delta_x.abs() > 1.0
        reward = torch.where(
            teleport_mask,
            torch.zeros_like(delta_x),
            delta_x * self.cfg.forward_reward_weight,
        )
        return reward

    def _check_termination_conditions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check for early termination conditions.

        Returns:
            Tuple of (terminated, truncated) boolean tensors.
            - terminated: Episode ended due to task completion/failure
            - truncated: Episode ended due to task-specific truncation
        """
        # No early termination conditions for now (can add collision detection, etc.)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros_like(terminated)

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specific environments.

        Args:
            env_ids: Indices of environments to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset Jetbot to default state
        root_state = self.scene["jetbot"].data.default_root_state[env_ids].clone()

        # Add environment origins (for parallel envs)
        root_state[:, :3] += self.scene.env_origins[env_ids]

        # Write root state to simulation
        self.scene["jetbot"].write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.scene["jetbot"].write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        # Reset joint states by directly assigning to the data buffers
        # This avoids the shape mismatch issue with write_joint_state_to_sim
        self.scene["jetbot"].data.joint_pos[env_ids] = self.scene["jetbot"].data.default_joint_pos[env_ids]
        self.scene["jetbot"].data.joint_vel[env_ids] = self.scene["jetbot"].data.default_joint_vel[env_ids]

        # Write the updated joint states to simulation
        self.scene["jetbot"].write_data_to_sim()

        # Reset task-specific state
        self.prev_x[env_ids] = root_state[:, 0]

        # Reset common navigation state from base class
        self._reset_navigation_state(env_ids)

        # Call parent reset (handles episode_length_buf, etc.)
        super()._reset_idx(env_ids)

    # Helper methods for action sampling (useful for testing)
    def sample_forward_action(self) -> torch.Tensor:
        """Sample action for moving forward.

        Returns:
            Action tensor for forward motion. Shape: (num_envs, 2)
        """
        # Forward at 0.8 to match isaac_sim_test.py
        # With action_scale=5.0, this gives 4.0 rad/s per wheel
        # (DifferentialController with [0.8, 0.0] produces ~3.8 rad/s)
        return torch.ones(self.num_envs, 2, device=self.device) * 0.8

    def sample_turn_action(self) -> torch.Tensor:
        """Sample action for turning (differential steering).

        Returns:
            Action tensor for turning. Shape: (num_envs, 2)
        """
        actions = torch.zeros(self.num_envs, 2, device=self.device)
        # Turn action to match isaac_sim_test.py differential steering
        # DifferentialController with [0.0, 1.0] produces [-1.875, 1.875] rad/s
        # With action_scale=5.0, we use 0.4 to get 2.0 rad/s differential
        actions[:, 0] = -0.4   # Left wheel backward
        actions[:, 1] = 0.4    # Right wheel forward
        return actions
