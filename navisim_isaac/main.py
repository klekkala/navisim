#!/usr/bin/env python3
"""
NaviSim-Isaac Main Entry Point

This script provides a unified interface for running navigation simulations
with Isaac Sim. It handles initialization, configuration, and execution of
different navigation scenarios.

Usage:
    # Simple test with default robot
    python main.py --mode simple

    # Navigation with waypoints
    python main.py --mode navigation --waypoints "[(2,0,0.5), (2,2,0.5), (0,2,0.5), (0,0,0.5)]"

    # Load sequence graph and navigate
    python main.py --mode sequence_graph --graph-path assets/sequence_graph.gpickle

    # Run in headless mode (no GUI)
    python main.py --mode navigation --headless

    # Custom configuration
    python main.py --config config/custom_config.yaml
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# NaviSim-Isaac imports
from navisim_isaac.isaac.simulator import IsaacSimulator
from navisim_isaac.isaac.navigation_robot import NavigationRobot
from navisim_isaac.isaac.navigation_controller import NavigationController
from navisim_isaac.isaac.terrain import TerrainManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NaviSimIsaacApp:
    """
    Main application class for NaviSim-Isaac integration.

    Orchestrates:
    - IsaacSim initialization
    - Scene and terrain loading
    - Robot spawning and control
    - Navigation execution
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize NaviSim-Isaac application.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()

        # Core components
        self.simulator: Optional[IsaacSimulator] = None
        self.robot: Optional[NavigationRobot] = None
        self.controller: Optional[NavigationController] = None
        self.terrain_manager: Optional[TerrainManager] = None

        # State
        self.is_initialized = False
        self.world = None
        self.stage = None

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'physics_dt': 1.0 / 60.0,      # 60 Hz physics
            'rendering_dt': 1.0 / 30.0,    # 30 Hz rendering
            'max_linear_speed': 1.0,        # m/s
            'max_angular_speed': 1.0,       # rad/s
            'position_tolerance': 0.2,      # meters
            'orientation_tolerance': 0.1,   # radians
            'robot_type': 'differential_drive',
            'robot_name': 'nav_robot',
            'initial_position': (0.0, 0.0, 0.5),
            'initial_orientation': (0.0, 0.0, 0.0, 1.0),
        }

    def initialize(self, headless: bool = False) -> None:
        """
        Initialize all components.

        Args:
            headless: Whether to run without GUI
        """
        logger.info("=" * 70)
        logger.info("NaviSim-Isaac Initialization")
        logger.info("=" * 70)

        # Step 1: Initialize IsaacSim
        logger.info("\n[1/4] Initializing IsaacSim World...")
        self.simulator = IsaacSimulator(config=self.config)
        self.simulator.initialize(headless=headless)

        self.world = self.simulator.get_world()
        self.stage = self.simulator.get_stage()
        logger.info("✓ IsaacSim World initialized")

        # Step 2: Initialize terrain manager
        logger.info("\n[2/4] Setting up Terrain Manager...")
        if self.stage is not None:
            self.terrain_manager = TerrainManager(stage=self.stage)
            logger.info("✓ Terrain Manager ready")
        else:
            logger.warning("! Stage not available, skipping terrain manager")

        # Step 3: Spawn robot
        logger.info("\n[3/4] Spawning Navigation Robot...")
        self.robot = NavigationRobot(
            robot_type=self.config.get('robot_type', 'differential_drive'),
            name=self.config.get('robot_name', 'nav_robot'),
            world=self.world
        )

        initial_pos = self.config.get('initial_position', (0.0, 0.0, 0.5))
        initial_orn = self.config.get('initial_orientation', (0.0, 0.0, 0.0, 1.0))

        self.robot.spawn(position=initial_pos, orientation=initial_orn)
        logger.info(f"✓ Robot spawned at {initial_pos}")

        # Step 4: Initialize navigation controller
        logger.info("\n[4/4] Initializing Navigation Controller...")
        self.controller = NavigationController(
            max_linear_speed=self.config.get('max_linear_speed', 1.0),
            max_angular_speed=self.config.get('max_angular_speed', 1.0),
            position_tolerance=self.config.get('position_tolerance', 0.2),
            orientation_tolerance=self.config.get('orientation_tolerance', 0.1)
        )
        logger.info("✓ Navigation Controller initialized")

        self.is_initialized = True
        logger.info("\n" + "=" * 70)
        logger.info("Initialization Complete!")
        logger.info("=" * 70)

    def load_sector_usd(
        self,
        usd_path: str,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        enable_physics: bool = True
    ) -> None:
        """
        Load a sector USD scene with terrain.

        Args:
            usd_path: Path to USD file
            position: World position offset
            enable_physics: Whether to enable physics collisions
        """
        if not self.is_initialized:
            raise RuntimeError("App not initialized. Call initialize() first.")

        logger.info(f"\nLoading sector USD from: {usd_path}")

        # Load USD scene
        scene_prim_path = self.simulator.load_sector_usd(usd_path, position)
        logger.info(f"✓ Loaded scene at: {scene_prim_path}")

        # Load terrain with physics
        if self.terrain_manager and enable_physics:
            try:
                terrain_prim = self.terrain_manager.load_terrain_from_usd(
                    usd_scene_path=scene_prim_path,
                    heightmap_prim_name="Heightmap",
                    enable_physics=True
                )
                self.terrain_manager.set_terrain_properties(
                    friction=0.8,
                    restitution=0.1
                )
                logger.info(f"✓ Terrain physics enabled: {terrain_prim}")
            except Exception as e:
                logger.warning(f"! Could not enable terrain physics: {e}")

    def set_waypoints(self, waypoints: List[Tuple[float, float, float]]) -> None:
        """
        Set navigation waypoints.

        Args:
            waypoints: List of (x, y, z) positions
        """
        if not self.is_initialized:
            raise RuntimeError("App not initialized. Call initialize() first.")

        self.controller.set_waypoint_path(waypoints)
        logger.info(f"✓ Set {len(waypoints)} waypoints for navigation")

    def run_navigation(
        self,
        max_steps: int = 3000,
        print_interval: int = 60
    ) -> None:
        """
        Run navigation simulation.

        Args:
            max_steps: Maximum simulation steps
            print_interval: Print progress every N steps
        """
        if not self.is_initialized:
            raise RuntimeError("App not initialized. Call initialize() first.")

        logger.info("\n" + "=" * 70)
        logger.info("Starting Navigation")
        logger.info("=" * 70)
        logger.info("Press Ctrl+C to stop\n")

        # Reset world to initialize physics
        self.world.reset()
        self.simulator.play()

        try:
            step_count = 0

            while step_count < max_steps:
                # Get current robot state
                position, orientation = self.robot.get_pose()

                # Compute velocity commands
                linear_vel, angular_vel = self.controller.compute_path_following_velocity(
                    current_position=position,
                    current_orientation=orientation
                )

                # Apply velocity to robot
                self.robot.set_velocity(linear_vel, angular_vel)

                # Step simulation
                self.simulator.step(num_steps=1)
                step_count += 1

                # Print progress
                if step_count % print_interval == 0:
                    waypoint_idx = self.controller.waypoint_index
                    total_waypoints = len(self.controller.waypoint_path)
                    logger.info(
                        f"Step {step_count:4d}: Pos=({position[0]:5.2f}, {position[1]:5.2f}), "
                        f"Waypoint={waypoint_idx}/{total_waypoints}, "
                        f"Vel=({linear_vel:4.2f}, {angular_vel:4.2f})"
                    )

                # Check if path complete
                if self.controller.current_waypoint is None:
                    logger.info(f"\n✓ Navigation complete after {step_count} steps!")
                    break

        except KeyboardInterrupt:
            logger.info("\n\nNavigation interrupted by user")

        finally:
            # Cleanup
            self.robot.stop()
            self.simulator.pause()
            logger.info("\n✓ Simulation paused")

    def run_simple_test(self, steps: int = 300) -> None:
        """
        Run a simple navigation test to a single waypoint.

        Args:
            steps: Number of simulation steps
        """
        if not self.is_initialized:
            raise RuntimeError("App not initialized. Call initialize() first.")

        logger.info("\n" + "=" * 70)
        logger.info("Running Simple Navigation Test")
        logger.info("=" * 70)

        # Set single waypoint
        target = (2.0, 2.0, 0.5)
        self.controller.set_waypoint(target)
        logger.info(f"Target: {target}\n")

        # Reset and play
        self.world.reset()
        self.simulator.play()

        try:
            for i in range(steps):
                pos, orn = self.robot.get_pose()
                lin_vel, ang_vel = self.controller.compute_velocity_to_point(
                    pos, orn, target
                )
                self.robot.set_velocity(lin_vel, ang_vel)
                self.simulator.step()

                if i % 60 == 0:
                    distance = np.linalg.norm(pos[:2] - np.array(target[:2]))
                    logger.info(
                        f"Step {i:3d}: Pos=({pos[0]:5.2f}, {pos[1]:5.2f}), "
                        f"Dist={distance:5.2f}m, Vel=({lin_vel:4.2f}, {ang_vel:4.2f})"
                    )

                # Check if reached
                distance = np.linalg.norm(pos[:2] - np.array(target[:2]))
                if distance < self.config['position_tolerance']:
                    logger.info(f"\n✓ Target reached after {i} steps!")
                    break

        except KeyboardInterrupt:
            logger.info("\n\nTest interrupted by user")

        finally:
            self.robot.stop()
            self.simulator.pause()
            logger.info("\n✓ Test complete")

    def shutdown(self) -> None:
        """Shutdown the application."""
        logger.info("\n" + "=" * 70)
        logger.info("Shutting down NaviSim-Isaac")
        logger.info("=" * 70)

        if self.robot:
            self.robot.stop()

        if self.simulator:
            self.simulator.shutdown()

        logger.info("✓ Shutdown complete")


def parse_waypoints(waypoint_str: str) -> List[Tuple[float, float, float]]:
    """
    Parse waypoints from string format.

    Args:
        waypoint_str: String like "[(0,0,0.5), (2,2,0.5)]"

    Returns:
        List of waypoint tuples
    """
    import ast
    try:
        waypoints = ast.literal_eval(waypoint_str)
        # Convert to list of tuples
        return [tuple(wp) if isinstance(wp, (list, tuple)) else wp
                for wp in waypoints]
    except Exception as e:
        logger.error(f"Failed to parse waypoints: {e}")
        return []


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NaviSim-Isaac: Navigation with Isaac Sim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple test
  python main.py --mode simple

  # Navigation with waypoints
  python main.py --mode navigation --waypoints "[(2,0,0.5), (2,2,0.5), (0,2,0.5)]"

  # Load sector USD
  python main.py --mode navigation --usd-path sector.usd --waypoints "[(5,0,0.5), (5,5,0.5)]"

  # Run headless
  python main.py --mode simple --headless
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['simple', 'navigation', 'sequence_graph'],
        default='simple',
        help='Simulation mode'
    )

    parser.add_argument(
        '--waypoints',
        type=str,
        help='Waypoints as string: "[(x1,y1,z1), (x2,y2,z2), ...]"'
    )

    parser.add_argument(
        '--usd-path',
        type=str,
        help='Path to sector USD file'
    )

    parser.add_argument(
        '--graph-path',
        type=str,
        help='Path to sequence graph pickle file'
    )

    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without GUI'
    )

    parser.add_argument(
        '--max-steps',
        type=int,
        default=3000,
        help='Maximum simulation steps'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom config YAML file'
    )

    args = parser.parse_args()

    # Load config
    config = None
    if args.config:
        logger.info(f"Loading config from: {args.config}")
        # TODO: Implement YAML config loading
        # config = load_yaml_config(args.config)

    # Create app
    app = NaviSimIsaacApp(config=config)

    try:
        # Initialize
        app.initialize(headless=args.headless)

        # Load USD if provided
        if args.usd_path:
            usd_path = Path(args.usd_path)
            if usd_path.exists():
                app.load_sector_usd(str(usd_path))
            else:
                logger.warning(f"USD file not found: {usd_path}")

        # Run based on mode
        if args.mode == 'simple':
            app.run_simple_test(steps=args.max_steps)

        elif args.mode == 'navigation':
            # Parse waypoints
            if args.waypoints:
                waypoints = parse_waypoints(args.waypoints)
                if waypoints:
                    app.set_waypoints(waypoints)
                    app.run_navigation(max_steps=args.max_steps)
                else:
                    logger.error("Invalid waypoints format")
            else:
                # Default square path
                default_waypoints = [
                    (2.0, 0.0, 0.5),
                    (2.0, 2.0, 0.5),
                    (0.0, 2.0, 0.5),
                    (0.0, 0.0, 0.5),
                ]
                logger.info("No waypoints provided, using default square path")
                app.set_waypoints(default_waypoints)
                app.run_navigation(max_steps=args.max_steps)

        elif args.mode == 'sequence_graph':
            if not args.graph_path:
                logger.error("--graph-path required for sequence_graph mode")
                return 1

            logger.info("Sequence graph mode not yet implemented")
            # TODO: Implement sequence graph navigation
            return 1

        logger.info("\n" + "=" * 70)
        logger.info("NaviSim-Isaac Session Complete!")
        logger.info("=" * 70)
        logger.info("To exit: Stop the simulation and close Isaac Sim")

        return 0

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        return 1

    finally:
        # Note: We don't call app.shutdown() here to keep IsaacSim running
        # Users can manually stop and close
        pass


if __name__ == "__main__":
    sys.exit(main())
