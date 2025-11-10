"""
Gymnasium Environment Module

Gymnasium-compatible environment for NaviSim-Isaac integration.
Each sector (node) from the sequence graph becomes a training environment.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NaviSimIsaacEnv(gym.Env):
    """
    Gymnasium environment for navigation training with NaviSim and IsaacSim.

    Each episode:
    1. Loads a sector's USD file (with Gaussian splat visual + heightmap physics)
    2. Spawns robot at start position
    3. Agent controls robot with velocity commands
    4. IsaacSim renders observations from Gaussian splat
    5. Heightmap physics detects collisions for termination
    6. Rewards progress toward goal

    Observation Space:
    - RGB image from camera (H x W x 3)
    - Robot state (position, orientation, velocity)

    Action Space:
    - Linear velocity (forward/backward)
    - Angular velocity (turn left/right)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        sequence_graph_path: str,
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the NaviSimIsaacEnv.

        Args:
            sequence_graph_path: Path to sequence graph pickle file
            config: Environment configuration dictionary
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()

        self.config = self._default_config()
        if config:
            self.config.update(config)

        self.render_mode = render_mode

        # Core components
        self.sequence_graph = None
        self.simulator = None
        self.robot = None
        self.terrain_manager = None
        self.current_sector = None

        # Episode state
        self.episode_step = 0
        self.max_episode_steps = self.config.get('max_episode_steps', 1000)
        self.start_position = None
        self.goal_position = None
        self.previous_distance_to_goal = None

        # Define action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Define observation space
        img_height = self.config.get('camera_height', 84)
        img_width = self.config.get('camera_width', 84)
        state_dim = 8  # [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, yaw]

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(img_height, img_width, 3),
                dtype=np.uint8
            ),
            'state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(state_dim,),
                dtype=np.float32
            )
        })

        # Initialize environment
        self._initialize(sequence_graph_path)

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'physics_dt': 1.0 / 60.0,
            'rendering_dt': 1.0 / 30.0,
            'max_episode_steps': 1000,
            'max_linear_speed': 1.0,
            'max_angular_speed': 1.0,
            'camera_height': 84,
            'camera_width': 84,
            'goal_tolerance': 0.5,  # meters
            'collision_penalty': -10.0,
            'goal_reward': 10.0,
            'step_penalty': -0.01,
            'distance_reward_scale': 1.0,
            'headless': True,  # Run without GUI for training
        }

    def _initialize(self, sequence_graph_path: str) -> None:
        """
        Initialize IsaacSim and load sequence graph.

        Args:
            sequence_graph_path: Path to sequence graph pickle
        """
        from navisim_isaac.isaac.simulator import IsaacSimulator
        from navisim_isaac.isaac.navigation_robot import NavigationRobot
        from navisim_isaac.isaac.terrain import TerrainManager
        from navisim_isaac.navisim.world.sequence_graph import SequenceGraph

        # Load sequence graph
        print(f"Loading sequence graph from: {sequence_graph_path}")
        self.sequence_graph = SequenceGraph(sequence_graph_path)

        # Initialize IsaacSim
        print("Initializing IsaacSim...")
        self.simulator = IsaacSimulator(config=self.config)
        self.simulator.initialize(headless=self.config['headless'])

        world = self.simulator.get_world()
        stage = self.simulator.get_stage()

        # Initialize terrain manager
        if stage is not None:
            self.terrain_manager = TerrainManager(stage=stage)

        # Create robot with camera
        camera_config = {
            'width': self.config['camera_width'],
            'height': self.config['camera_height'],
            'position': (0.3, 0.0, 0.2),  # Relative to robot center
            'orientation': (0.0, 0.0, 0.0, 1.0),  # Looking forward
            'horizontal_aperture': 20.955,  # Field of view
            'focal_length': 24.0
        }
        self.robot = NavigationRobot(
            robot_type='differential_drive',
            name='rl_robot',
            world=world,
            camera_config=camera_config
        )

        print("Environment initialized successfully")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (can specify 'sector_id')

        Returns:
            Tuple of (initial_observation, info_dict)
        """
        super().reset(seed=seed)

        # Select sector to load
        if options and 'sector_id' in options:
            sector_id = options['sector_id']
        else:
            # Randomly select a sector
            sector_ids = self.sequence_graph.get_sector_ids()
            sector_id = self.np_random.choice(sector_ids)

        # Load sector
        self._load_sector(sector_id)

        # Set start and goal positions
        self._set_start_goal_positions()

        # Spawn robot at start
        if not self.robot.is_spawned:
            self.robot.spawn(
                position=self.start_position,
                orientation=(0.0, 0.0, 0.0, 1.0)
            )
        else:
            self.robot.set_pose(
                position=self.start_position,
                orientation=(0.0, 0.0, 0.0, 1.0)
            )

        # Reset episode state
        self.episode_step = 0
        position, _ = self.robot.get_pose()
        self.previous_distance_to_goal = np.linalg.norm(
            position[:2] - np.array(self.goal_position[:2])
        )

        # Reset IsaacSim world
        self.simulator.get_world().reset()
        self.simulator.play()

        # Get initial observation
        observation = self._get_observation()

        info = {
            'sector_id': sector_id,
            'start_position': self.start_position,
            'goal_position': self.goal_position,
            'episode_step': self.episode_step
        }

        return observation, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: [linear_velocity, angular_velocity] normalized to [-1, 1]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Scale action to actual velocities
        linear_vel = action[0] * self.config['max_linear_speed']
        angular_vel = action[1] * self.config['max_angular_speed']

        # Apply action to robot
        self.robot.set_velocity(linear_vel, angular_vel)

        # Step simulation
        self.simulator.step(num_steps=1)
        self.episode_step += 1

        # Get current state
        position, orientation = self.robot.get_pose()

        # Check termination conditions
        terminated = False
        truncated = False
        info = {'episode_step': self.episode_step}

        # Check collision
        if self._check_collision():
            terminated = True
            info['termination_reason'] = 'collision'

        # Check goal reached
        distance_to_goal = np.linalg.norm(position[:2] - np.array(self.goal_position[:2]))
        if distance_to_goal < self.config['goal_tolerance']:
            terminated = True
            info['termination_reason'] = 'goal_reached'

        # Check timeout
        if self.episode_step >= self.max_episode_steps:
            truncated = True
            info['termination_reason'] = 'timeout'

        # Compute reward
        reward = self._compute_reward(
            position, distance_to_goal,
            terminated, info.get('termination_reason')
        )

        # Update distance tracking
        self.previous_distance_to_goal = distance_to_goal

        # Get observation
        observation = self._get_observation()

        info['distance_to_goal'] = distance_to_goal
        info['position'] = position
        info['reward'] = reward

        return observation, reward, terminated, truncated, info

    def _load_sector(self, sector_id: str) -> None:
        """
        Load a sector's USD scene into IsaacSim.

        Args:
            sector_id: Sector identifier
        """
        # Unload previous sector
        if self.current_sector is not None:
            self.current_sector.unload_all()

        # Get sector from graph
        self.current_sector = self.sequence_graph.get_sector(sector_id)

        print(f"Loading sector: {sector_id}")
        print(f"  USD path: {self.current_sector.usd_path}")

        # Check USD file exists
        if not self.current_sector.has_usd_file():
            raise FileNotFoundError(f"USD file not found: {self.current_sector.usd_path}")

        # Load USD scene into IsaacSim
        scene_prim_path = self.simulator.load_sector_usd(
            usd_path=self.current_sector.scene_path,
            position=(0.0, 0.0, 0.0)
        )

        # Enable physics on heightmap
        if self.terrain_manager:
            try:
                self.terrain_manager.load_terrain_from_usd(
                    usd_scene_path=scene_prim_path,
                    heightmap_prim_name="Heightmap",
                    enable_physics=True
                )
                self.terrain_manager.set_terrain_properties(
                    friction=0.8,
                    restitution=0.1
                )
                print(f"  Loaded heightmap physics for collision detection")
            except Exception as e:
                print(f"  Warning: Could not enable terrain physics: {e}")

        print(f"✓ Sector {sector_id} loaded")

    def _set_start_goal_positions(self) -> None:
        """
        Set start and goal positions for the episode.
        For now, uses fixed positions. Can be extended to sample from sector metadata.
        """
        # TODO: Extract start/goal from sector metadata or boundary
        # For now, use simple defaults
        self.start_position = (0.0, 0.0, 0.5)
        self.goal_position = (5.0, 5.0, 0.5)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current observation from the environment.

        Returns:
            Dictionary with 'image' and 'state'
        """
        # Get robot state
        position, orientation = self.robot.get_pose()

        # Convert quaternion to yaw angle
        yaw = self._quaternion_to_yaw(orientation)

        # State vector: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, yaw]
        state = np.array([
            position[0], position[1], position[2],
            orientation[0], orientation[1], orientation[2], orientation[3],
            yaw
        ], dtype=np.float32)

        # Get RGB image from camera
        # TODO: Implement camera rendering from Gaussian splat scene
        # For now, return placeholder
        image = self._render_camera()

        return {
            'image': image,
            'state': state
        }

    def _render_camera(self) -> np.ndarray:
        """
        Render RGB image from robot's camera.
        The camera captures the scene including the Gaussian splat visuals.

        Returns:
            RGB image array (H, W, 3)
        """
        if self.robot and self.robot.has_camera():
            # Get image from robot's camera
            image = self.robot.get_camera_image()

            if image is not None:
                return image

        # Fallback: return placeholder if camera not available
        height = self.config['camera_height']
        width = self.config['camera_width']
        return np.zeros((height, width, 3), dtype=np.uint8)

    def _check_collision(self) -> bool:
        """
        Check for collision at current pose using PhysX and heightmap.

        Returns:
            True if collision detected, False otherwise
        """
        # TODO: Implement proper collision detection using PhysX
        # This should check:
        # 1. Robot colliding with heightmap terrain
        # 2. Robot falling below terrain
        # 3. Out of sector bounds

        position, _ = self.robot.get_pose()

        # Simple collision check: if robot falls below ground
        if position[2] < -0.5:  # Below ground level
            return True

        # Check if out of bounds (simple box check)
        # TODO: Use sector boundary polygon
        if abs(position[0]) > 50 or abs(position[1]) > 50:
            return True

        return False

    def _compute_reward(
        self,
        position: np.ndarray,
        distance_to_goal: float,
        terminated: bool,
        termination_reason: Optional[str]
    ) -> float:
        """
        Compute reward for current state.

        Reward components:
        - Distance to goal (shaped reward)
        - Goal reached (large positive)
        - Collision (large negative)
        - Step penalty (small negative to encourage efficiency)

        Args:
            position: Robot position
            distance_to_goal: Current distance to goal
            terminated: Whether episode terminated
            termination_reason: Reason for termination

        Returns:
            Reward value
        """
        reward = 0.0

        # Step penalty
        reward += self.config['step_penalty']

        # Distance-based reward (progress toward goal)
        if self.previous_distance_to_goal is not None:
            distance_progress = self.previous_distance_to_goal - distance_to_goal
            reward += distance_progress * self.config['distance_reward_scale']

        # Terminal rewards
        if terminated:
            if termination_reason == 'goal_reached':
                reward += self.config['goal_reward']
            elif termination_reason == 'collision':
                reward += self.config['collision_penalty']

        return reward

    def _quaternion_to_yaw(self, quaternion: np.ndarray) -> float:
        """
        Extract yaw angle from quaternion.

        Args:
            quaternion: [x, y, z, w]

        Returns:
            Yaw angle in radians
        """
        x, y, z, w = quaternion
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Returns:
            RGB image if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "rgb_array":
            return self._render_camera()
        elif self.render_mode == "human":
            # IsaacSim already renders in its GUI
            return None
        return None

    def close(self) -> None:
        """
        Close the environment and cleanup resources.
        """
        print("Closing environment...")

        if self.current_sector:
            self.current_sector.unload_all()

        if self.sequence_graph:
            self.sequence_graph.unload_all_sectors()

        if self.robot:
            self.robot.stop()

        if self.simulator:
            self.simulator.shutdown()

        print("Environment closed")
