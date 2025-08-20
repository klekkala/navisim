"""Physics component for entity physics simulation."""

import numpy as np
from ..entity import Component


class PhysicsComponent(Component):
    """
    Physics component following SAPIEN patterns.
    
    Manages physical properties like mass, forces, and collision.
    """
    
    def __init__(
        self,
        mass: float = 1.0,
        friction: float = 0.5,
        restitution: float = 0.0
    ):
        """
        Initialize physics component.
        
        Args:
            mass: Entity mass
            friction: Friction coefficient
            restitution: Restitution coefficient
        """
        super().__init__()
        
        self.mass = mass
        self.friction = friction
        self.restitution = restitution
        
        # Forces and torques
        self.applied_force = np.zeros(3, dtype=np.float32)
        self.applied_torque = np.zeros(3, dtype=np.float32)
        
        # Collision state
        self.is_colliding = False
        self.collision_contacts = []
    
    def update(self, dt: float) -> None:
        """Update physics state."""
        # Apply forces to transform component if available
        if self.entity:
            from .transform import TransformComponent
            transform = self.entity.get_component(TransformComponent)
            if transform:
                # Simple force integration (F = ma)
                acceleration = self.applied_force / self.mass
                transform.linear_velocity += acceleration * dt
        
        # Clear applied forces
        self.applied_force.fill(0.0)
        self.applied_torque.fill(0.0)
    
    def apply_force(self, force: np.ndarray) -> None:
        """Apply force to the entity."""
        self.applied_force += force
    
    def apply_torque(self, torque: np.ndarray) -> None:
        """Apply torque to the entity."""
        self.applied_torque += torque
    
    def set_mass(self, mass: float) -> None:
        """Set entity mass."""
        self.mass = max(mass, 1e-6)  # Prevent zero mass
    
    def reset(self) -> None:
        """Reset physics component."""
        self.applied_force.fill(0.0)
        self.applied_torque.fill(0.0)
        self.is_colliding = False
        self.collision_contacts.clear()
    
    def __repr__(self) -> str:
        return f"PhysicsComponent(mass={self.mass}, friction={self.friction})"