"""Navigation task for Navisim."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional

from ..base import BaseEnv, register_env
from ...agents.navigation_agent import NavigationAgent
from ...assets.loaders.sector import Sector
from ...simulation.rendering.camera import NavisimCamera


@register_env("NavisimNavigation-v1", max_episode_steps=1000)
class NavigationTask(BaseEnv):
    """
    Navigation task environment following ManiSkill patterns.
    
    Task: Navigate an agent to a goal position while avoiding obstacles.
    Uses single-environment Isaac Sim simulation with Gaussian Splatting rendering.
    """
    
    def __init__(
        self,
        scene_path: Optional[str] = None,
        goal_radius: float = 0.5,
        max_speed: float = 2.0,
        **kwargs
    ):
        self.scene_path = scene_path
        self.goal_radius = goal_radius
        self.max_speed = max_speed
        
        super().__init__(**kwargs)
    
    def _build_world(self):
        """Build the navigation world."""
        # Load scene
        if self.scene_path:
            self.sector = Sector(self.scene_path)
            self.sim.load_scene(self.sector)
        
        # Create agent
        self.agent = NavigationAgent(
            scene=self.sim.scene,
            max_speed=self.max_speed
        )
        
        # Setup camera for rendering
        self.camera = NavisimCamera(
            scene=self.sim.scene,
            width=640,
            height=480
        )
        
        # Initialize goal position
        self._sample_goal()
    
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-self.max_speed, -np.pi]),
            high=np.array([self.max_speed, np.pi]),
            dtype=np.float32
        )
        
        # Observation space depends on obs_mode
        if self.obs_mode == "state_dict":
            self.observation_space = spaces.Dict({
                "agent_pos": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                "agent_vel": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                "goal_pos": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                "goal_distance": spaces.Box(0, np.inf, (1,), dtype=np.float32)
            })
        elif self.obs_mode == "rgb":
            self.observation_space = spaces.Box(
                0, 255, (480, 640, 3), dtype=np.uint8
            )
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")
    
    def _sample_goal(self):
        """Sample a random goal position."""
        # Simple random goal in a reasonable range
        self.goal_pos = np.random.uniform(-10, 10, size=3)
        self.goal_pos[2] = 0.5  # Fixed height
    
    def _get_obs(self) -> Dict[str, Any]:
        """Get current observations."""
        agent_pos = self.agent.get_position()
        agent_vel = self.agent.get_velocity()
        goal_distance = np.linalg.norm(agent_pos - self.goal_pos)
        
        if self.obs_mode == "state_dict":
            return {
                "agent_pos": agent_pos.astype(np.float32),
                "agent_vel": agent_vel.astype(np.float32),
                "goal_pos": self.goal_pos.astype(np.float32),
                "goal_distance": np.array([goal_distance], dtype=np.float32)
            }
        elif self.obs_mode == "rgb":
            return self.camera.get_rgb()
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dict."""
        agent_pos = self.agent.get_position()
        goal_distance = np.linalg.norm(agent_pos - self.goal_pos)
        
        return {
            "goal_distance": goal_distance,
            "success": goal_distance < self.goal_radius,
            "episode_step": self._episode_step
        }
    
    def _reset_simulation(self):
        """Reset the simulation state."""
        # Reset agent position
        start_pos = np.array([0.0, 0.0, 0.5])
        self.agent.set_position(start_pos)
        self.agent.set_velocity(np.zeros(3))
        
        # Sample new goal
        self._sample_goal()
    
    def _apply_action(self, action):
        """Apply action to the agent."""
        linear_vel = action[0]
        angular_vel = action[1]
        self.agent.set_velocity([linear_vel, 0.0, 0.0])
        self.agent.set_angular_velocity([0.0, 0.0, angular_vel])
    
    def _step_simulation(self):
        """Step the simulation forward."""
        self.sim.step()
    
    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        agent_pos = self.agent.get_position()
        goal_distance = np.linalg.norm(agent_pos - self.goal_pos)
        
        # Distance-based reward with goal bonus
        reward = -goal_distance * 0.1
        
        # Success bonus
        if goal_distance < self.goal_radius:
            reward += 10.0
        
        # Collision penalty (simplified)
        if agent_pos[2] < 0.1:  # Below ground
            reward -= 5.0
        
        return float(reward)
    
    def _check_terminated(self) -> bool:
        """Check if episode is terminated."""
        agent_pos = self.agent.get_position()
        goal_distance = np.linalg.norm(agent_pos - self.goal_pos)
        
        # Success condition
        if goal_distance < self.goal_radius:
            return True
        
        # Failure conditions
        if agent_pos[2] < 0.1:  # Below ground
            return True
        
        return False