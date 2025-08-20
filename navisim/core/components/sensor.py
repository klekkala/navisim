"""Sensor component for perception and data collection."""

import numpy as np
from typing import Optional, Dict, Any, List
from ..entity import Component


class SensorComponent(Component):
    """
    Sensor component following SAPIEN patterns.
    
    Manages sensors like cameras, IMU, and other perception devices.
    """
    
    def __init__(
        self,
        sensor_type: str = "camera",
        update_frequency: float = 60.0,
        sensor_config: Dict[str, Any] = None
    ):
        """
        Initialize sensor component.
        
        Args:
            sensor_type: Type of sensor (camera, imu, lidar, etc.)
            update_frequency: Sensor update frequency in Hz
            sensor_config: Sensor-specific configuration
        """
        super().__init__()
        
        self.sensor_type = sensor_type
        self.update_frequency = update_frequency
        self.sensor_config = sensor_config or {}
        
        # Sensor state
        self.is_active = True
        self.last_update_time = 0.0
        self.sensor_data: Optional[np.ndarray] = None
        self.data_history: List[np.ndarray] = []
        
        # Update interval based on frequency
        self.update_interval = 1.0 / update_frequency if update_frequency > 0 else 0.0
    
    def update(self, dt: float) -> None:
        """Update sensor and collect data."""
        if not self.is_active:
            return
        
        self.last_update_time += dt
        
        # Update based on sensor frequency
        if self.update_interval == 0.0 or self.last_update_time >= self.update_interval:
            self._collect_sensor_data()
            self.last_update_time = 0.0
    
    def _collect_sensor_data(self) -> None:
        """Collect sensor data (to be overridden by specific sensor types)."""
        # Base implementation - specific sensors should override this
        if self.sensor_type == "camera":
            self._collect_camera_data()
        elif self.sensor_type == "imu":
            self._collect_imu_data()
        # Add more sensor types as needed
    
    def _collect_camera_data(self) -> None:
        """Collect camera sensor data."""
        # Placeholder for camera data collection
        # This would interface with the rendering system
        width = self.sensor_config.get("width", 640)
        height = self.sensor_config.get("height", 480)
        channels = self.sensor_config.get("channels", 3)
        
        # Placeholder data
        self.sensor_data = np.zeros((height, width, channels), dtype=np.uint8)
    
    def _collect_imu_data(self) -> None:
        """Collect IMU sensor data."""
        # Placeholder for IMU data collection
        # This would interface with the physics system
        if self.entity:
            transform = self.entity.get_component("TransformComponent")
            if transform:
                # IMU data: linear acceleration + angular velocity
                self.sensor_data = np.concatenate([
                    transform.linear_velocity,  # Linear acceleration (simplified)
                    transform.angular_velocity  # Angular velocity
                ])
    
    def get_latest_data(self) -> Optional[np.ndarray]:
        """Get the most recent sensor data."""
        return self.sensor_data.copy() if self.sensor_data is not None else None
    
    def get_data_history(self, num_frames: int = 10) -> List[np.ndarray]:
        """Get recent sensor data history."""
        return self.data_history[-num_frames:] if self.data_history else []
    
    def set_active(self, active: bool) -> None:
        """Set sensor active state."""
        self.is_active = active
    
    def set_update_frequency(self, frequency: float) -> None:
        """Set sensor update frequency."""
        self.update_frequency = frequency
        self.update_interval = 1.0 / frequency if frequency > 0 else 0.0
    
    def reset(self) -> None:
        """Reset sensor component."""
        self.last_update_time = 0.0
        self.sensor_data = None
        self.data_history.clear()
    
    def __repr__(self) -> str:
        return f"SensorComponent(type={self.sensor_type}, freq={self.update_frequency}Hz, active={self.is_active})"