"""
Navigation Robot Module

Defines the physical robot agent for navigation tasks in IsaacSim 5.0.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np

# IsaacSim 5.0 imports
# Note: Camera is imported later in _initialize_camera() after SimulationApp is ready
try:
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.objects import DynamicCuboid
    from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False
    print("Warning: IsaacSim not available. Robot will run in simulation mode.")


class NavigationRobot:
    """
    Physical robot agent for navigation in IsaacSim.

    This class handles:
    - Robot spawning and initialization
    - Pose management (position and orientation)
    - Velocity control (linear and angular)
    - Sensor data collection
    - Physics interaction with environment
    """

    def __init__(
        self,
        robot_type: str = "differential_drive",
        name: str = "nav_robot",
        world=None,
        camera_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize NavigationRobot.

        Args:
            robot_type: Type of robot ("differential_drive", "ackermann", "omnidirectional")
            name: Unique name for the robot instance
            world: IsaacSim World instance
            camera_config: Camera configuration dictionary
        """
        self.robot_type = robot_type
        self.name = name
        self.world = world
        self.prim_path = None
        self.is_spawned = False
        self.robot_instance = None
        self.controller = None
        self.camera = None

        # Robot state
        self.position = np.array([0.0, 0.0, 0.0])  # (x, y, z)
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion (x, y, z, w)
        self.linear_velocity = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])

        # Robot parameters
        self.wheel_radius = 0.1  # meters
        self.wheel_base = 0.5    # meters (distance between wheels)
        self.max_linear_speed = 2.0   # m/s
        self.max_angular_speed = 2.0  # rad/s

        # Camera configuration
        self.camera_config = camera_config or {
            'width': 84,
            'height': 84,
            'position': (0.3, 0.0, 0.2),  # Relative to robot center
            'orientation': (0.0, 0.0, 0.0, 1.0),  # Looking forward
            'horizontal_aperture': 20.955,  # Field of view
            'focal_length': 24.0
        }

    def spawn(
        self,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.5),
        orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        usd_path: Optional[str] = None
    ) -> str:
        """
        Spawn the robot in the scene using IsaacSim 5.0.

        Args:
            position: Initial position (x, y, z)
            orientation: Initial orientation as quaternion (x, y, z, w)
            usd_path: Optional path to custom robot USD file

        Returns:
            Prim path of spawned robot
        """
        self.position = np.array(position)
        self.orientation = np.array(orientation)
        self.prim_path = f"/World/{self.name}"

        if not ISAAC_SIM_AVAILABLE:
            print(f"[Simulation Mode] Spawning robot '{self.name}' at position {position}")
            self.is_spawned = True
            return self.prim_path

        # Real IsaacSim implementation
        if usd_path:
            # Load custom robot from USD
            self.robot_instance = Robot(
                prim_path=self.prim_path,
                name=self.name,
                usd_path=usd_path,
                position=position,
                orientation=orientation
            )
        else:
            # Use Jetbot as default wheeled robot
            from omni.isaac.wheeled_robots.robots import Jetbot
            self.robot_instance = Jetbot(
                prim_path=self.prim_path,
                name=self.name,
                position=position,
                orientation=orientation
            )

        # Initialize controller for differential drive
        if self.robot_type == "differential_drive" and ISAAC_SIM_AVAILABLE:
            self.controller = DifferentialController(
                name=f"{self.name}_controller",
                wheel_radius=self.wheel_radius,
                wheel_base=self.wheel_base
            )

        # Add to world scene
        if self.world:
            self.world.scene.add(self.robot_instance)

        # Mark as spawned BEFORE initializing camera (camera init checks this flag)
        self.is_spawned = True

        # Initialize camera attached to robot
        self._initialize_camera()

        print(f"Spawned robot '{self.name}' at position {position}")
        print(f"Robot type: {self.robot_type}")
        print(f"Prim path: {self.prim_path}")

        return self.prim_path

    def spawn_simple_robot(
        self,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.5),
        size: Tuple[float, float, float] = (0.5, 0.5, 0.3)
    ) -> str:
        """
        Spawn a simple box-shaped robot for testing with IsaacSim 5.0.

        Args:
            position: Initial position (x, y, z)
            size: Robot dimensions (length, width, height)

        Returns:
            Prim path of spawned robot
        """
        self.position = np.array(position)
        self.prim_path = f"/World/{self.name}"

        if not ISAAC_SIM_AVAILABLE:
            print(f"[Simulation Mode] Spawning simple robot at {position} with size {size}")
            self.is_spawned = True
            return self.prim_path

        # Real IsaacSim implementation
        self.robot_instance = DynamicCuboid(
            prim_path=self.prim_path,
            name=self.name,
            position=position,
            size=np.array(size),
            color=np.array([0.2, 0.6, 1.0]),
            mass=10.0
        )

        # Add to world scene
        if self.world:
            self.world.scene.add(self.robot_instance)

        # Mark as spawned BEFORE initializing camera
        self.is_spawned = True

        # Initialize camera attached to robot
        self._initialize_camera()

        print(f"Spawned simple robot at {position} with size {size}")
        return self.prim_path

    def set_velocity(
        self,
        linear_velocity: float,
        angular_velocity: float
    ) -> None:
        """
        Set robot velocity using IsaacSim 5.0 (differential drive control).

        Args:
            linear_velocity: Forward velocity in m/s
            angular_velocity: Angular velocity in rad/s
        """
        # Clamp velocities to max limits
        linear_velocity = np.clip(linear_velocity, -self.max_linear_speed, self.max_linear_speed)
        angular_velocity = np.clip(angular_velocity, -self.max_angular_speed, self.max_angular_speed)

        self.linear_velocity[0] = linear_velocity  # Forward direction
        self.angular_velocity[2] = angular_velocity  # Yaw rotation

        if not ISAAC_SIM_AVAILABLE or self.robot_instance is None:
            return

        # Real IsaacSim implementation
        try:
            if self.controller is not None:
                # Use differential controller to compute wheel velocities
                wheel_velocities = self.controller.forward(
                    command=[linear_velocity, angular_velocity]
                )
                # Apply wheel velocities to robot
                if hasattr(self.robot_instance, 'apply_wheel_actions'):
                    self.robot_instance.apply_wheel_actions(wheel_velocities)
            else:
                # Direct velocity control for simple robots (DynamicCuboid)
                # Get current orientation to compute forward direction
                _, orientation = self.get_pose()

                # Convert linear velocity to world space based on robot orientation
                # Extract yaw from quaternion
                x, y, z, w = orientation
                yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

                # Compute velocity in world space
                vel_x = linear_velocity * np.cos(yaw)
                vel_y = linear_velocity * np.sin(yaw)

                # Apply velocities
                if hasattr(self.robot_instance, 'set_linear_velocity'):
                    self.robot_instance.set_linear_velocity(
                        np.array([vel_x, vel_y, 0.0])
                    )
                if hasattr(self.robot_instance, 'set_angular_velocity'):
                    self.robot_instance.set_angular_velocity(
                        np.array([0.0, 0.0, angular_velocity])
                    )
        except Exception as e:
            print(f"Warning: Failed to set velocity: {e}")

    def set_wheel_velocities(
        self,
        left_wheel_velocity: float,
        right_wheel_velocity: float
    ) -> None:
        """
        Set individual wheel velocities (for differential drive).

        Args:
            left_wheel_velocity: Left wheel velocity in rad/s
            right_wheel_velocity: Right wheel velocity in rad/s
        """
        # Convert wheel velocities to robot velocities
        linear_vel = self.wheel_radius * (left_wheel_velocity + right_wheel_velocity) / 2.0
        angular_vel = self.wheel_radius * (right_wheel_velocity - left_wheel_velocity) / self.wheel_base

        self.set_velocity(linear_vel, angular_vel)

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current robot pose using IsaacSim 5.0.

        Returns:
            Tuple of (position, orientation)
            - position: (x, y, z) numpy array
            - orientation: (x, y, z, w) quaternion numpy array
        """
        if ISAAC_SIM_AVAILABLE and self.robot_instance is not None:
            position, orientation = self.robot_instance.get_world_pose()
            self.position = np.array(position)
            self.orientation = np.array(orientation)

        return self.position.copy(), self.orientation.copy()

    def set_pose(
        self,
        position: Optional[Tuple[float, float, float]] = None,
        orientation: Optional[Tuple[float, float, float, float]] = None
    ) -> None:
        """
        Set robot pose directly using IsaacSim 5.0 (teleportation).

        Args:
            position: Target position (x, y, z), None to keep current
            orientation: Target orientation quaternion (x, y, z, w), None to keep current
        """
        if position is not None:
            self.position = np.array(position)

        if orientation is not None:
            self.orientation = np.array(orientation)

        if ISAAC_SIM_AVAILABLE and self.robot_instance is not None:
            self.robot_instance.set_world_pose(
                position=self.position if position is not None else None,
                orientation=self.orientation if orientation is not None else None
            )

        print(f"Setting robot pose - Position: {self.position}, Orientation: {self.orientation}")

    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current robot velocity.

        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """
        # In real IsaacSim:
        # if hasattr(self, 'robot_instance'):
        #     self.linear_velocity = self.robot_instance.get_linear_velocity()
        #     self.angular_velocity = self.robot_instance.get_angular_velocity()

        return self.linear_velocity.copy(), self.angular_velocity.copy()

    def stop(self) -> None:
        """
        Stop the robot (set all velocities to zero).
        """
        self.set_velocity(0.0, 0.0)

    def get_state(self) -> Dict[str, Any]:
        """
        Get complete robot state.

        Returns:
            Dictionary containing robot state information
        """
        position, orientation = self.get_pose()
        linear_vel, angular_vel = self.get_velocity()

        return {
            "position": position,
            "orientation": orientation,
            "linear_velocity": linear_vel,
            "angular_velocity": angular_vel,
            "is_spawned": self.is_spawned,
            "prim_path": self.prim_path
        }

    def remove(self) -> None:
        """
        Remove robot from the scene.
        """
        if self.is_spawned and self.prim_path:
            # In real IsaacSim:
            # if hasattr(self, 'robot_instance'):
            #     self.stage.RemovePrim(self.prim_path)

            print(f"Removing robot from scene: {self.prim_path}")
            self.is_spawned = False
            self.prim_path = None

    def _initialize_camera(self) -> None:
        """
        Initialize RGB camera sensor attached to the robot.
        The camera is mounted on the robot to capture first-person view images.

        IMPORTANT: Camera must be imported AFTER SimulationApp is initialized.
        """
        if not ISAAC_SIM_AVAILABLE or not self.is_spawned:
            print("[Simulation Mode] Camera initialization skipped")
            return

        # Camera prim path (attached to robot)
        camera_prim_path = f"{self.prim_path}/Camera"

        try:
            # Import Camera AFTER SimulationApp is ready (not at top level)
            from omni.isaac.sensor import Camera

            print(f"Initializing camera at: {camera_prim_path}")

            # Create camera sensor
            self.camera = Camera(
                prim_path=camera_prim_path,
                name=f"{self.name}_camera",
                position=np.array(self.camera_config['position']),
                orientation=np.array(self.camera_config['orientation']),
                frequency=30,  # 30 Hz capture rate
                resolution=(self.camera_config['width'], self.camera_config['height'])
            )

            # Set camera properties
            self.camera.set_horizontal_aperture(self.camera_config['horizontal_aperture'])
            self.camera.set_focal_length(self.camera_config['focal_length'])

            # Initialize camera (start data collection)
            self.camera.initialize()

            print(f"✓ Camera initialized successfully!")
            print(f"  Resolution: {self.camera_config['width']}x{self.camera_config['height']}")
            print(f"  Position (relative): {self.camera_config['position']}")

        except ImportError as e:
            print(f"Warning: Could not import Camera module: {e}")
            print(f"  Make sure SimulationApp is initialized first")
            self.camera = None
        except Exception as e:
            print(f"Warning: Could not initialize camera: {e}")
            import traceback
            traceback.print_exc()
            self.camera = None

    def get_camera_image(self) -> Optional[np.ndarray]:
        """
        Get RGB image from the robot's camera.

        Returns:
            RGB image as numpy array (H, W, 3) with dtype uint8, or None if unavailable
        """
        if self.camera is None:
            # Return placeholder in simulation mode or if camera not initialized
            height = self.camera_config['height']
            width = self.camera_config['width']
            return np.zeros((height, width, 3), dtype=np.uint8)

        try:
            # Get RGB data from camera
            # The camera returns RGBA, we extract RGB
            frame = self.camera.get_rgba()

            if frame is None:
                print("Warning: Camera returned None frame")
                return None

            # Convert to uint8 and extract RGB channels
            rgb_image = (frame[:, :, :3] * 255).astype(np.uint8)

            return rgb_image

        except Exception as e:
            print(f"Warning: Failed to capture camera image: {e}")
            import traceback
            traceback.print_exc()
            return None

    def has_camera(self) -> bool:
        """
        Check if robot has an active camera.

        Returns:
            True if camera is initialized and available
        """
        return self.camera is not None

    def __repr__(self) -> str:
        """String representation of robot."""
        return f"NavigationRobot(name='{self.name}', type='{self.robot_type}', spawned={self.is_spawned})"
