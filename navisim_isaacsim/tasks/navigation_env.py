# tasks/navigation_env.py

"""IsaacLab RL environment for Jetbot navigation in warehouse."""

import torch
import gymnasium as gym
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene
from configs.navigation_env_cfg import NavisimNavigationEnvCfg


class NavisimNavigationEnv(DirectRLEnv):
    """
    IsaacLab RL environment for Jetbot navigation in warehouse.

    **Observations**: Jetbot root state (13D: position, orientation, velocities)

    **Actions**: Wheel velocities (2D: left_wheel, right_wheel) in range [-1, 1]

    **Rewards**: Forward progress along x-axis
    """

    cfg: NavisimNavigationEnvCfg

    def __init__(self, cfg: NavisimNavigationEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the navigation environment.

        Args:
            cfg: Configuration for the environment.
            render_mode: Render mode for the environment. Defaults to None.
            **kwargs: Additional keyword arguments passed to DirectRLEnv.
        """
        super().__init__(cfg, render_mode, **kwargs)

        # Task-specific state tensors (on GPU)
        self.prev_x = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # Note: Action and observation spaces are automatically created by DirectRLEnv
        # from cfg.action_space and cfg.observation_space dimensions

    def _setup_scene(self):
        """Setup the scene with Jetbot robot and warehouse."""
        # Create scene from config
        self.scene = InteractiveScene(self.cfg.scene)

        # Clone environments (create num_envs copies)
        self.scene.clone_environments(copy_from_source=False)

        # Add scene entities to simulation
        self.sim.reset()

        # Set camera view
        self.sim.set_camera_view(
            eye=self.cfg.viewer_eye,
            target=self.cfg.viewer_target,
        )

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

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on forward progress.

        Returns:
            Reward tensor for each environment. Shape: (num_envs,)
        """
        # Current x position
        curr_x = self.scene["jetbot"].data.root_state_w[:, 0]

        # Reward = forward displacement
        reward = (curr_x - self.prev_x) * self.cfg.forward_reward_weight

        # Update previous position
        self.prev_x = curr_x.clone()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine episode termination.

        Returns:
            Tuple of (terminated, truncated) boolean tensors.
            - terminated: Episode ended due to task completion/failure
            - truncated: Episode ended due to time limit
        """
        # Calculate max episode steps
        max_episode_steps = int(self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))

        # Episode truncated when max length reached
        time_out = self.episode_length_buf >= max_episode_steps - 1

        # No early termination conditions for now (can add collision detection, etc.)
        terminated = torch.zeros_like(time_out, dtype=torch.bool)

        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specific environments.

        Args:
            env_ids: Indices of environments to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Number of environments to reset
        num_resets = len(env_ids)

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

        # Call parent reset (handles episode_length_buf, etc.)
        super()._reset_idx(env_ids)

    def _set_debug_vis(self, debug_vis: bool) -> None:
        """Enable/disable debug visualization.

        Args:
            debug_vis: Whether to enable debug visualization.
        """
        # Optional: Add debug visualizations here (e.g., goal markers, trajectory lines)
        pass

    # Helper methods for action sampling (useful for testing)
    def sample_forward_action(self) -> torch.Tensor:
        """Sample action for moving forward.

        Returns:
            Action tensor for forward motion. Shape: (num_envs, 2)
        """
        return torch.ones(self.num_envs, 2, device=self.device) * 0.8

    def sample_turn_action(self) -> torch.Tensor:
        """Sample action for turning (differential steering).

        Returns:
            Action tensor for turning. Shape: (num_envs, 2)
        """
        actions = torch.zeros(self.num_envs, 2, device=self.device)
        actions[:, 0] = 0.5   # Left wheel forward
        actions[:, 1] = -0.5  # Right wheel backward
        return actions
