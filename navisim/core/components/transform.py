"""Transform component for pose and motion state."""

import numpy as np
from ..entity import Component


class TransformComponent(Component):
    """
    Transform component following SAPIEN patterns.
    
    Manages entity pose, velocity, and acceleration in world space.
    """
    
    def __init__(
        self,
        pose: np.ndarray = None,
        linear_velocity: np.ndarray = None,
        angular_velocity: np.ndarray = None
    ):
        """
        Initialize transform component.
        
        Args:
            pose: 4x4 transformation matrix (world_T_entity)
            linear_velocity: 3D linear velocity
            angular_velocity: 3D angular velocity
        """
        super().__init__()
        
        # Pose as 4x4 transformation matrix
        self.pose = pose if pose is not None else np.eye(4, dtype=np.float64)
        
        # Velocities
        self.linear_velocity = linear_velocity if linear_velocity is not None else np.zeros(3, dtype=np.float32)
        self.angular_velocity = angular_velocity if angular_velocity is not None else np.zeros(3, dtype=np.float32)
        
        # Store initial state for reset
        self._initial_pose = self.pose.copy()
        self._initial_linear_velocity = self.linear_velocity.copy()
        self._initial_angular_velocity = self.angular_velocity.copy()
    
    def update(self, dt: float) -> None:
        """Update transform based on velocities."""
        # Simple integration (can be overridden for more sophisticated integration)
        translation = self.linear_velocity * dt
        self.pose[:3, 3] += translation
    
    def set_pose(self, pose: np.ndarray) -> None:
        """Set entity pose."""
        self.pose = pose.copy()
    
    def set_position(self, position: np.ndarray) -> None:
        """Set entity position."""
        self.pose[:3, 3] = position
    
    def set_rotation_matrix(self, rotation: np.ndarray) -> None:
        """Set entity rotation matrix."""
        self.pose[:3, :3] = rotation
    
    def get_position(self) -> np.ndarray:
        """Get entity position."""
        return self.pose[:3, 3].copy()
    
    def get_rotation_matrix(self) -> np.ndarray:
        """Get entity rotation matrix."""
        return self.pose[:3, :3].copy()
    
    def translate(self, translation: np.ndarray) -> None:
        """Apply translation to entity."""
        self.pose[:3, 3] += translation
    
    def set_linear_velocity(self, velocity: np.ndarray) -> None:
        """Set linear velocity."""
        self.linear_velocity = velocity.copy()
    
    def set_angular_velocity(self, velocity: np.ndarray) -> None:
        """Set angular velocity."""
        self.angular_velocity = velocity.copy()
    
    def reset(self) -> None:
        """Reset transform to initial state."""
        self.pose = self._initial_pose.copy()
        self.linear_velocity = self._initial_linear_velocity.copy()
        self.angular_velocity = self._initial_angular_velocity.copy()
    
    def __repr__(self) -> str:
        pos = self.get_position()
        return f"TransformComponent(pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}])"