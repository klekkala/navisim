"""
Navigation environment implementation following SAPIEN patterns.
"""

import uuid
import random
from typing import Dict, Any, Optional

import gymnasium as gym
import numpy as np

from .base_env import NavisimEnv, AgentState, RenderFrame
from ..core.scene import NavisimScene
from ..core.entity import Entity
from ..core.components.transform import TransformComponent
from ..simulation.physics.bridge import NavisimBridgeClient
from ..assets.loaders.sequence_graph import SequenceGraph
from ..utils.spaces import RenderMode


class NavigationEnv(NavisimEnv):
    """
    Navigation environment following SAPIEN patterns.
    
    Implements the abstract methods from NavisimEnv for navigation tasks.
    Uses entity-component system for modular design.
    """
    
    def __init__(
        self,
        sequence_graph: SequenceGraph,
        render_mode: str = "rgb_array",
        control_freq: int = 60,
        **kwargs
    ):
        """
        Initialize navigation environment.
        
        Args:
            sequence_graph: Graph of navigation sequences
            render_mode: Rendering mode
            control_freq: Control frequency
        """
        self.sequence_graph = sequence_graph
        self._current_sequence_id: Optional[str] = None
        self._sequence_index = 0
        self._target_locations = np.zeros((0, 2), dtype=np.float32)
        
        super().__init__(
            control_freq=control_freq,
            render_mode=render_mode,
            **kwargs
        )
    
    def _build_world(self) -> None:
        """Build the simulation world following SAPIEN patterns."""
        # Create scene
        self._scene = NavisimScene(timestep=self.timestep)
        
        # Create bridge for physics simulation
        self._bridge = NavisimBridgeClient()
        
        # Create agent entity with transform component
        self._agent_entity = Entity(name="Agent")
        self._agent_entity.add_component(TransformComponent())
        self._scene.add_entity(self._agent_entity)
    
    def _setup_spaces(self) -> None:
        """Setup action and observation spaces."""
        # Action space: discrete actions for navigation
        self.action_space = gym.spaces.Discrete(4)  # stop, forward, left, right
        
        # Observation space: RGB image + pose + velocities
        self.observation_space = gym.spaces.Dict({
            "rgb": gym.spaces.Box(low=0, high=255, shape=(720, 1280, 3), dtype=np.uint8),
            "pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float64),
            "linear_velocity": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "angular_velocity": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        # Get transform component from agent entity
        transform = self._agent_entity.get_component(TransformComponent)
        
        return {
            "rgb": self._render_frame.rgb if self._render_frame.rgb is not None else np.zeros((720, 1280, 3), dtype=np.uint8),
            "pose": transform.pose.copy(),
            "linear_velocity": transform.linear_velocity.copy(),
            "angular_velocity": transform.angular_velocity.copy(),
        }
    
    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        # Simple distance-based reward
        transform = self._agent_entity.get_component(TransformComponent)
        agent_pos = transform.get_position()[:2]  # x, y only
        
        if self._target_locations.size > 0:
            # Distance to closest target
            distances = np.linalg.norm(self._target_locations - agent_pos, axis=1)
            closest_distance = np.min(distances)
            
            # Reward for being close to target
            if closest_distance < 0.5:
                return 1.0  # Success reward
            else:
                return -0.01  # Small step penalty
        
        return -0.01  # Default step penalty
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        transform = self._agent_entity.get_component(TransformComponent)
        agent_pos = transform.get_position()[:2]
        
        if self._target_locations.size > 0:
            distances = np.linalg.norm(self._target_locations - agent_pos, axis=1)
            return np.min(distances) < 0.5  # Terminate when close to target
        
        return False
    
    def _reset_simulation(self) -> None:
        """Reset simulation state."""
        # Select random sequence
        sequence_ids = self.sequence_graph.get_sequence_ids()
        self._current_sequence_id = random.choice(sequence_ids)
        sequence = self.sequence_graph.get_sequence(self._current_sequence_id)
        
        # Load sector
        current_sector = sequence[self._sequence_index]
        self._scene.set_active_sector(current_sector)
        
        # Reset agent to random spawn pose
        spawn_pose = self._scene.get_random_spawn_pose()
        transform = self._agent_entity.get_component(TransformComponent)
        transform.set_pose(spawn_pose)
        transform.set_linear_velocity(np.zeros(3, dtype=np.float32))
        transform.set_angular_velocity(np.zeros(3, dtype=np.float32))
        
        # Get target locations
        self._target_locations = self._scene.get_target_locations()
        
        # Reset scene
        self._scene.reset()
    
    def _apply_action(self, action: int) -> None:
        """Apply action to simulation."""
        # Convert discrete action to velocity command
        action_map = {
            0: {"linear": [0.0, 0.0, 0.0], "angular": [0.0, 0.0, 0.0]},  # stop
            1: {"linear": [1.0, 0.0, 0.0], "angular": [0.0, 0.0, 0.0]},  # forward
            2: {"linear": [0.0, 0.0, 0.0], "angular": [0.0, 0.0, 0.5]},  # turn left
            3: {"linear": [0.0, 0.0, 0.0], "angular": [0.0, 0.0, -0.5]}, # turn right
        }
        
        action_cmd = action_map.get(action, action_map[0])
        
        # Send to bridge if available
        if self._bridge:
            try:
                # Get camera parameters
                camera_params = {
                    "width": 1280,
                    "height": 720,
                    "fx": 900.0,
                    "fy": 900.0,
                    "cx": 640.0,
                    "cy": 360.0,
                }
                
                # Get sector info
                sector_id = f"{self._current_sequence_id}/0"  # Simplified
                version = "v0"
                
                # Step simulation
                header, color, depth = self._bridge.step_render(
                    action={"mode": "vel", **action_cmd},
                    dt=self.timestep,
                    camera=camera_params,
                    sector_id=sector_id,
                    version=version
                )
                
                # Update render frame
                self._render_frame.rgb = color
                self._render_frame.depth = depth
                self._render_frame.render_time_ms = header.get("render_ms", 0.0)
                
            except Exception as e:
                # Fallback to local simulation if bridge fails
                print(f"Bridge communication failed: {e}")
                self._apply_action_local(action_cmd)
        else:
            self._apply_action_local(action_cmd)
    
    def _apply_action_local(self, action_cmd: Dict[str, Any]) -> None:
        """Apply action locally without bridge."""
        # Update agent transform directly
        transform = self._agent_entity.get_component(TransformComponent)
        transform.set_linear_velocity(np.array(action_cmd["linear"], dtype=np.float32))
        transform.set_angular_velocity(np.array(action_cmd["angular"], dtype=np.float32))
    
    def _step_simulation(self) -> None:
        """Step physics simulation."""
        # Step the scene (which updates all entities)
        self._scene.step()
    
    def _update_agent_state(self) -> None:
        """Update agent state from simulation."""
        # Agent state is maintained in the transform component
        transform = self._agent_entity.get_component(TransformComponent)
        
        # Update global agent state for compatibility
        self._agent_state.pose_world_T_agent = transform.pose.copy()
        self._agent_state.linear_velocity = transform.linear_velocity.copy()
        self._agent_state.angular_velocity = transform.angular_velocity.copy()
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        info = super()._get_info()
        
        # Add navigation-specific info
        transform = self._agent_entity.get_component(TransformComponent)
        agent_pos = transform.get_position()[:2]
        
        if self._target_locations.size > 0:
            distances = np.linalg.norm(self._target_locations - agent_pos, axis=1)
            closest_distance = float(np.min(distances))
        else:
            closest_distance = float("inf")
        
        info.update({
            "distance_to_target": closest_distance,
            "num_targets": len(self._target_locations),
            "current_sequence": self._current_sequence_id,
            "render_time_ms": self._render_frame.render_time_ms,
        })
        
        return info