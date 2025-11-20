# policies/scripted_policies.py

"""Scripted (rule-based) navigation policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import math

from .base_policy import BasePolicy, PolicyConfig


@dataclass
class ScriptedPolicyConfig(PolicyConfig):
    """Configuration for scripted policies."""

    action_scale: float = 0.8  # Scale for actions (0-1)


class ForwardPolicy(BasePolicy):
    """Simple policy that drives forward."""

    def __init__(self, config: ScriptedPolicyConfig, num_envs: int, observation_space: Any, action_space: Any):
        super().__init__(config, num_envs, observation_space, action_space)
        self.action_scale = config.action_scale

    def compute_action(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Drive forward with both wheels at equal speed."""
        actions = torch.ones(self.num_envs, 2, device=self.device) * self.action_scale
        return actions


class CirclePolicy(BasePolicy):
    """Policy that drives in circles using differential steering."""

    def __init__(self, config: ScriptedPolicyConfig, num_envs: int, observation_space: Any, action_space: Any):
        super().__init__(config, num_envs, observation_space, action_space)
        self.action_scale = config.action_scale
        self.radius_factor = 0.5  # Controls circle tightness

    def compute_action(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Drive in circle: left wheel faster than right."""
        actions = torch.zeros(self.num_envs, 2, device=self.device)
        actions[:, 0] = self.action_scale  # Left wheel
        actions[:, 1] = self.action_scale * self.radius_factor  # Right wheel (slower)
        return actions


class RandomPolicy(BasePolicy):
    """Random exploration policy."""

    def __init__(self, config: ScriptedPolicyConfig, num_envs: int, observation_space: Any, action_space: Any):
        super().__init__(config, num_envs, observation_space, action_space)
        self.action_scale = config.action_scale

    def compute_action(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample random actions."""
        actions = torch.rand(self.num_envs, 2, device=self.device) * 2 - 1  # Range [-1, 1]
        actions = actions * self.action_scale
        return actions


@dataclass
class WaypointPolicyConfig(ScriptedPolicyConfig):
    """Configuration for waypoint following policy."""

    forward_speed: float = 0.8
    turn_speed: float = 0.5
    waypoint_threshold: float = 0.5  # Distance to consider waypoint reached
    action_pattern: str = "forward_turn"  # "forward_turn" or "spiral"
    forward_steps: int = 100
    turn_steps: int = 20


class WaypointPolicy(BasePolicy):
    """
    Policy that follows predefined waypoints or patterns.

    Can execute different motion patterns:
    - forward_turn: Move forward, then turn
    - spiral: Gradually increasing turn radius
    """

    def __init__(self, config: WaypointPolicyConfig, num_envs: int, observation_space: Any, action_space: Any):
        super().__init__(config, num_envs, observation_space, action_space)
        self.forward_speed = config.forward_speed
        self.turn_speed = config.turn_speed
        self.pattern = config.action_pattern
        self.forward_steps = config.forward_steps
        self.turn_steps = config.turn_steps

        # Per-environment step counters
        self.env_step_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def compute_action(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute actions based on motion pattern."""
        actions = torch.zeros(self.num_envs, 2, device=self.device)

        if self.pattern == "forward_turn":
            # Cycle: forward for N steps, turn for M steps
            cycle_length = self.forward_steps + self.turn_steps
            phase = self.env_step_count % cycle_length

            # Forward phase
            forward_mask = phase < self.forward_steps
            actions[forward_mask, :] = self.forward_speed

            # Turn phase
            turn_mask = ~forward_mask
            actions[turn_mask, 0] = self.turn_speed   # Left wheel
            actions[turn_mask, 1] = -self.turn_speed  # Right wheel (opposite)

        elif self.pattern == "spiral":
            # Gradually increase turn rate
            t = self.env_step_count.float() / 100.0
            turn_factor = torch.sin(t * 0.1).unsqueeze(1)

            actions[:, 0] = self.forward_speed  # Left wheel constant
            actions[:, 1] = self.forward_speed * (1 - 0.5 * turn_factor.squeeze())  # Right wheel varies

        # Increment step counter
        self.env_step_count += 1

        return actions

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Reset step counters for specified environments."""
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.env_step_count[env_ids] = 0


@dataclass
class ObstacleAvoidancePolicyConfig(ScriptedPolicyConfig):
    """Configuration for reactive obstacle avoidance."""

    forward_speed: float = 0.8
    turn_speed: float = 0.6
    safe_distance: float = 1.0  # Minimum safe distance from obstacles


class ObstacleAvoidancePolicy(BasePolicy):
    """
    Reactive obstacle avoidance policy using observations.

    Uses robot's position and orientation to avoid boundaries.
    """

    def __init__(self, config: ObstacleAvoidancePolicyConfig, num_envs: int,
                 observation_space: Any, action_space: Any):
        super().__init__(config, num_envs, observation_space, action_space)
        self.forward_speed = config.forward_speed
        self.turn_speed = config.turn_speed
        self.safe_distance = config.safe_distance

    def compute_action(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute actions based on robot state.

        Observations (root_state_w): [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        """
        obs = observations["policy"]  # Shape: (num_envs, 13)

        # Extract position and orientation
        pos = obs[:, :3]  # (x, y, z)
        # quat = obs[:, 3:7]  # (w, x, y, z)

        actions = torch.zeros(self.num_envs, 2, device=self.device)

        # Simple boundary avoidance: turn away if too far from origin
        distance_from_origin = torch.norm(pos[:, :2], dim=1)  # 2D distance

        # If far from origin, turn back
        turn_mask = distance_from_origin > self.safe_distance

        # Forward when safe
        actions[~turn_mask, :] = self.forward_speed

        # Turn when too far
        actions[turn_mask, 0] = self.turn_speed
        actions[turn_mask, 1] = -self.turn_speed * 0.5

        return actions
