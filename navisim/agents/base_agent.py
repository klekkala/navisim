"""Base agent class following ManiSkill patterns."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..core.entity import Entity
from ..core.components import TransformComponent, PhysicsComponent


class BaseAgent(ABC):
    """
    Base agent class following ManiSkill patterns.
    
    Provides common functionality for all agent types:
    - Position and velocity control
    - State management
    - Action interface
    """
    
    def __init__(
        self,
        scene: Any,
        name: str = "agent",
        initial_pos: Optional[np.ndarray] = None,
        max_speed: float = 2.0,
        **kwargs
    ):
        self.scene = scene
        self.name = name
        self.max_speed = max_speed
        
        # Create entity with required components
        self.entity = Entity(name)
        self.entity.add_component(TransformComponent())
        self.entity.add_component(PhysicsComponent())
        
        # Initialize position
        if initial_pos is not None:
            self.set_position(initial_pos)
        else:
            self.set_position(np.zeros(3))
        
        # Initialize state
        self._velocity = np.zeros(3)
        self._angular_velocity = np.zeros(3)
    
    @abstractmethod
    def _setup_agent(self):
        """Setup agent-specific configuration. Override in subclasses."""
        pass
    
    def get_position(self) -> np.ndarray:
        """Get current position."""
        transform = self.entity.get_component(TransformComponent)
        return transform.position.copy()
    
    def set_position(self, position: np.ndarray):
        """Set position."""
        transform = self.entity.get_component(TransformComponent)
        transform.position = position.copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity."""
        return self._velocity.copy()
    
    def set_velocity(self, velocity: np.ndarray):
        """Set velocity."""
        self._velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        
        # Update physics component
        physics = self.entity.get_component(PhysicsComponent)
        physics.velocity = self._velocity.copy()
    
    def get_angular_velocity(self) -> np.ndarray:
        """Get current angular velocity."""
        return self._angular_velocity.copy()
    
    def set_angular_velocity(self, angular_velocity: np.ndarray):
        """Set angular velocity."""
        self._angular_velocity = angular_velocity.copy()
        
        # Update physics component
        physics = self.entity.get_component(PhysicsComponent)
        physics.angular_velocity = self._angular_velocity.copy()
    
    def get_orientation(self) -> np.ndarray:
        """Get current orientation (quaternion)."""
        transform = self.entity.get_component(TransformComponent)
        return transform.rotation.copy()
    
    def set_orientation(self, quaternion: np.ndarray):
        """Set orientation (quaternion)."""
        transform = self.entity.get_component(TransformComponent)
        transform.rotation = quaternion.copy()
    
    def apply_force(self, force: np.ndarray):
        """Apply force to the agent."""
        physics = self.entity.get_component(PhysicsComponent)
        if not hasattr(physics, 'forces'):
            physics.forces = []
        physics.forces.append(force.copy())
    
    def apply_torque(self, torque: np.ndarray):
        """Apply torque to the agent."""
        physics = self.entity.get_component(PhysicsComponent)
        if not hasattr(physics, 'torques'):
            physics.torques = []
        physics.torques.append(torque.copy())
    
    def step(self, dt: float):
        """Step the agent physics."""
        # Update position based on velocity
        current_pos = self.get_position()
        new_pos = current_pos + self._velocity * dt
        self.set_position(new_pos)
        
        # Update orientation based on angular velocity
        if np.linalg.norm(self._angular_velocity) > 1e-6:
            # Simple euler integration for rotation
            # In practice, you'd use proper quaternion integration
            pass
    
    def reset(self, initial_pos: Optional[np.ndarray] = None):
        """Reset agent to initial state."""
        if initial_pos is not None:
            self.set_position(initial_pos)
        else:
            self.set_position(np.zeros(3))
        
        self.set_velocity(np.zeros(3))
        self.set_angular_velocity(np.zeros(3))
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get agent state as dictionary."""
        return {
            "position": self.get_position(),
            "velocity": self.get_velocity(),
            "orientation": self.get_orientation(),
            "angular_velocity": self.get_angular_velocity()
        }
    
    def set_state_dict(self, state: Dict[str, Any]):
        """Set agent state from dictionary."""
        if "position" in state:
            self.set_position(state["position"])
        if "velocity" in state:
            self.set_velocity(state["velocity"])
        if "orientation" in state:
            self.set_orientation(state["orientation"])
        if "angular_velocity" in state:
            self.set_angular_velocity(state["angular_velocity"])