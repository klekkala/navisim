"""Navigation agent for autonomous navigation tasks."""

import numpy as np
from typing import Any, Dict, Optional

from .base_agent import BaseAgent
from ..core.components import SensorComponent


class NavigationAgent(BaseAgent):
    """
    Navigation agent with sensors and navigation capabilities.
    
    Features:
    - LIDAR/depth sensing
    - GPS positioning
    - IMU orientation
    - Path planning capabilities
    """
    
    def __init__(
        self,
        scene: Any,
        name: str = "nav_agent",
        sensor_range: float = 10.0,
        sensor_resolution: int = 64,
        **kwargs
    ):
        self.sensor_range = sensor_range
        self.sensor_resolution = sensor_resolution
        
        super().__init__(scene, name, **kwargs)
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup navigation-specific sensors and capabilities."""
        # Add sensor component
        sensor_comp = SensorComponent()
        sensor_comp.sensor_type = "lidar"
        sensor_comp.range = self.sensor_range
        sensor_comp.resolution = self.sensor_resolution
        self.entity.add_component(sensor_comp)
        
        # Initialize sensor data
        self._lidar_data = np.full(self.sensor_resolution, self.sensor_range)
        self._gps_data = np.zeros(3)
        self._imu_data = np.zeros(6)  # [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    
    def get_lidar_scan(self) -> np.ndarray:
        """Get LIDAR scan data."""
        # In real implementation, this would interface with simulation
        # For now, return simulated data
        return self._lidar_data.copy()
    
    def get_gps_position(self) -> np.ndarray:
        """Get GPS position."""
        # GPS is just the current position with some noise
        pos = self.get_position()
        noise = np.random.normal(0, 0.1, 3)  # 10cm GPS noise
        return pos + noise
    
    def get_imu_data(self) -> np.ndarray:
        """Get IMU data (acceleration and angular velocity)."""
        # In real implementation, this would come from physics simulation
        # For now, return simulated data based on current motion
        vel = self.get_velocity()
        ang_vel = self.get_angular_velocity()
        
        # Simple acceleration estimate (derivative of velocity)
        if not hasattr(self, '_prev_velocity'):
            self._prev_velocity = vel
        
        dt = 0.02  # Assume 50Hz update rate
        accel = (vel - self._prev_velocity) / dt
        self._prev_velocity = vel.copy()
        
        return np.concatenate([accel, ang_vel])
    
    def get_sensor_data(self) -> Dict[str, np.ndarray]:
        """Get all sensor data."""
        return {
            "lidar": self.get_lidar_scan(),
            "gps": self.get_gps_position(), 
            "imu": self.get_imu_data(),
            "position": self.get_position(),
            "velocity": self.get_velocity(),
            "orientation": self.get_orientation()
        }
    
    def navigate_to_goal(self, goal_pos: np.ndarray, dt: float = 0.02) -> np.ndarray:
        """
        Simple navigation to goal position.
        Returns desired velocity command.
        """
        current_pos = self.get_position()
        goal_direction = goal_pos - current_pos
        goal_distance = np.linalg.norm(goal_direction)
        
        if goal_distance < 0.1:  # Close enough
            return np.zeros(3)
        
        # Normalize direction
        goal_direction_norm = goal_direction / goal_distance
        
        # Simple proportional control
        desired_speed = min(self.max_speed, goal_distance * 2.0)
        desired_velocity = goal_direction_norm * desired_speed
        
        return desired_velocity
    
    def avoid_obstacles(self, desired_velocity: np.ndarray) -> np.ndarray:
        """
        Simple obstacle avoidance using LIDAR data.
        Modifies desired velocity to avoid obstacles.
        """
        lidar_data = self.get_lidar_scan()
        
        # Find minimum distance in forward direction
        forward_indices = slice(self.sensor_resolution // 4, 3 * self.sensor_resolution // 4)
        forward_distances = lidar_data[forward_indices]
        min_forward_distance = np.min(forward_distances)
        
        # If obstacle too close, modify velocity
        safety_distance = 1.0
        if min_forward_distance < safety_distance:
            # Reduce forward velocity
            speed_reduction = 1.0 - (min_forward_distance / safety_distance)
            desired_velocity *= (1.0 - speed_reduction)
            
            # Add side velocity to avoid obstacle
            if len(forward_distances) > 0:
                left_distance = np.mean(lidar_data[:self.sensor_resolution // 4])
                right_distance = np.mean(lidar_data[3 * self.sensor_resolution // 4:])
                
                if left_distance > right_distance:
                    # Go left
                    desired_velocity[1] += 0.5 * speed_reduction
                else:
                    # Go right  
                    desired_velocity[1] -= 0.5 * speed_reduction
        
        return desired_velocity
    
    def navigate_with_avoidance(self, goal_pos: np.ndarray, dt: float = 0.02) -> np.ndarray:
        """
        Navigate to goal with obstacle avoidance.
        Returns final velocity command.
        """
        # Get desired velocity to goal
        desired_vel = self.navigate_to_goal(goal_pos, dt)
        
        # Apply obstacle avoidance
        safe_vel = self.avoid_obstacles(desired_vel)
        
        return safe_vel
    
    def update_sensors(self):
        """Update sensor readings from simulation."""
        # In real implementation, this would query the simulation
        # for updated sensor data
        
        # Simulate LIDAR scan
        pos = self.get_position()
        for i in range(self.sensor_resolution):
            angle = (i / self.sensor_resolution) * 2 * np.pi
            # Simple ray casting simulation
            ray_distance = self.sensor_range  # Default max range
            self._lidar_data[i] = ray_distance
        
        # Update GPS (current position)
        self._gps_data = self.get_position()
    
    def step(self, dt: float):
        """Step the navigation agent."""
        # Update sensors
        self.update_sensors()
        
        # Call parent step
        super().step(dt)
    
    def reset(self, initial_pos: Optional[np.ndarray] = None):
        """Reset navigation agent."""
        super().reset(initial_pos)
        
        # Reset sensor data
        self._lidar_data.fill(self.sensor_range)
        self._gps_data = self.get_position()
        self._imu_data.fill(0.0)
        
        if hasattr(self, '_prev_velocity'):
            delattr(self, '_prev_velocity')