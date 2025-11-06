"""
Navigation Controller Module

Provides navigation control algorithms for the robot.
"""

from typing import Tuple, Optional, List
import numpy as np


class NavigationController:
    """
    Navigation controller for robot movement.

    Implements various navigation strategies:
    - Waypoint following
    - Path tracking
    - Obstacle avoidance
    - Velocity control
    """

    def __init__(
        self,
        max_linear_speed: float = 2.0,
        max_angular_speed: float = 2.0,
        position_tolerance: float = 0.1,
        orientation_tolerance: float = 0.1
    ):
        """
        Initialize NavigationController.

        Args:
            max_linear_speed: Maximum linear velocity (m/s)
            max_angular_speed: Maximum angular velocity (rad/s)
            position_tolerance: Tolerance for reaching target position (m)
            orientation_tolerance: Tolerance for reaching target orientation (rad)
        """
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance

        # Controller gains
        self.k_linear = 1.0    # Proportional gain for linear velocity
        self.k_angular = 2.0   # Proportional gain for angular velocity

        # Current waypoint
        self.current_waypoint = None
        self.waypoint_path: List[Tuple[float, float]] = []
        self.waypoint_index = 0

    def compute_velocity_to_point(
        self,
        current_position: np.ndarray,
        current_orientation: np.ndarray,
        target_position: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        """
        Compute velocities to reach a target point.

        Args:
            current_position: Current robot position (x, y, z)
            current_orientation: Current orientation as quaternion (x, y, z, w)
            target_position: Target position (x, y, z)

        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """
        # Extract 2D position (ignore z for navigation)
        current_pos_2d = current_position[:2]
        target_pos_2d = np.array(target_position[:2])

        # Compute distance and angle to target
        delta = target_pos_2d - current_pos_2d
        distance = np.linalg.norm(delta)

        if distance < self.position_tolerance:
            # Reached target
            return 0.0, 0.0

        # Compute target angle
        target_angle = np.arctan2(delta[1], delta[0])

        # Get current yaw from quaternion
        current_yaw = self.quaternion_to_yaw(current_orientation)

        # Compute angle difference
        angle_diff = self.normalize_angle(target_angle - current_yaw)

        # Proportional control
        linear_vel = self.k_linear * distance
        angular_vel = self.k_angular * angle_diff

        # Slow down linear velocity if we need to turn
        if abs(angle_diff) > np.pi / 4:  # 45 degrees
            linear_vel *= 0.5

        # Clamp to max speeds
        linear_vel = np.clip(linear_vel, 0.0, self.max_linear_speed)
        angular_vel = np.clip(angular_vel, -self.max_angular_speed, self.max_angular_speed)

        return float(linear_vel), float(angular_vel)

    def compute_velocity_to_orientation(
        self,
        current_orientation: np.ndarray,
        target_yaw: float
    ) -> Tuple[float, float]:
        """
        Compute velocities to reach a target orientation.

        Args:
            current_orientation: Current orientation as quaternion (x, y, z, w)
            target_yaw: Target yaw angle (radians)

        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """
        current_yaw = self.quaternion_to_yaw(current_orientation)
        angle_diff = self.normalize_angle(target_yaw - current_yaw)

        if abs(angle_diff) < self.orientation_tolerance:
            # Reached target orientation
            return 0.0, 0.0

        # Only angular velocity for pure rotation
        angular_vel = self.k_angular * angle_diff
        angular_vel = np.clip(angular_vel, -self.max_angular_speed, self.max_angular_speed)

        return 0.0, float(angular_vel)

    def set_waypoint(self, waypoint: Tuple[float, float, float]) -> None:
        """
        Set a single waypoint target.

        Args:
            waypoint: Target waypoint (x, y, z)
        """
        self.current_waypoint = waypoint
        print(f"Setting waypoint: {waypoint}")

    def set_waypoint_path(self, waypoints: List[Tuple[float, float, float]]) -> None:
        """
        Set a path of waypoints to follow.

        Args:
            waypoints: List of waypoints [(x, y, z), ...]
        """
        self.waypoint_path = waypoints
        self.waypoint_index = 0
        if waypoints:
            self.current_waypoint = waypoints[0]
        print(f"Setting waypoint path with {len(waypoints)} waypoints")

    def get_next_waypoint(
        self,
        current_position: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get next waypoint in path if current one is reached.

        Args:
            current_position: Current robot position

        Returns:
            Next waypoint or None if path complete
        """
        if not self.waypoint_path or self.waypoint_index >= len(self.waypoint_path):
            return None

        # Check if current waypoint reached
        current_wp = self.waypoint_path[self.waypoint_index]
        distance = np.linalg.norm(current_position[:2] - np.array(current_wp[:2]))

        if distance < self.position_tolerance:
            # Move to next waypoint
            self.waypoint_index += 1
            if self.waypoint_index < len(self.waypoint_path):
                self.current_waypoint = self.waypoint_path[self.waypoint_index]
                print(f"Reached waypoint {self.waypoint_index}, moving to next")
                return self.current_waypoint
            else:
                print("Path complete!")
                self.current_waypoint = None
                return None

        return self.current_waypoint

    def compute_path_following_velocity(
        self,
        current_position: np.ndarray,
        current_orientation: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute velocities for following the waypoint path.

        Args:
            current_position: Current robot position
            current_orientation: Current robot orientation

        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """
        # Update waypoint if current one reached
        waypoint = self.get_next_waypoint(current_position)

        if waypoint is None:
            return 0.0, 0.0

        # Compute velocity to current waypoint
        return self.compute_velocity_to_point(
            current_position,
            current_orientation,
            waypoint
        )

    @staticmethod
    def quaternion_to_yaw(quaternion: np.ndarray) -> float:
        """
        Extract yaw angle from quaternion.

        Args:
            quaternion: Quaternion (x, y, z, w)

        Returns:
            Yaw angle in radians
        """
        x, y, z, w = quaternion
        # Yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    @staticmethod
    def yaw_to_quaternion(yaw: float) -> np.ndarray:
        """
        Convert yaw angle to quaternion (rotation around Z-axis).

        Args:
            yaw: Yaw angle in radians

        Returns:
            Quaternion (x, y, z, w)
        """
        return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)])

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        Normalize angle to [-pi, pi].

        Args:
            angle: Angle in radians

        Returns:
            Normalized angle
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def reset(self) -> None:
        """
        Reset controller state.
        """
        self.current_waypoint = None
        self.waypoint_path = []
        self.waypoint_index = 0
