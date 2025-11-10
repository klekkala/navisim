"""
Navigation Example with IsaacSim 5.0

Demonstrates how to:
1. Initialize IsaacSim world
2. Load a sector USD scene
3. Spawn a navigation robot
4. Control the robot to navigate waypoints
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# NaviSim-Isaac imports
from navisim_isaac.isaac.simulator import IsaacSimulator
from navisim_isaac.isaac.navigation_robot import NavigationRobot
from navisim_isaac.isaac.navigation_controller import NavigationController
from navisim_isaac.isaac.terrain import TerrainManager

# Try to import PIL for image saving
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install Pillow")


def main():
    """
    Main navigation example demonstrating robot placement and navigation.
    """
    print("=" * 60)
    print("NaviSim-Isaac Navigation Example")
    print("=" * 60)

    # Configuration
    config = {
        'physics_dt': 1.0 / 60.0,  # 60 Hz physics
        'rendering_dt': 1.0 / 30.0  # 30 Hz rendering
    }

    # Step 1: Initialize IsaacSim
    print("\n[Step 1] Initializing IsaacSim...")
    simulator = IsaacSimulator(config=config)
    simulator.initialize(headless=False)

    # Get world and stage
    world = simulator.get_world()
    stage = simulator.get_stage()

    # Check if IsaacSim is properly initialized
    if world is None:
        print("\n" + "=" * 60)
        print("ERROR: IsaacSim is not available!")
        print("=" * 60)
        print("\nThis example requires Isaac Sim to run.")
        print("\nTo run this example:")
        print("1. Install Isaac Sim from: https://developer.nvidia.com/isaac-sim")
        print("2. Run using Isaac Sim's Python:")
        print("   ~/.local/share/ov/pkg/isaac_sim-*/python.sh navisim_isaac/examples/navigation_example.py")
        print("\nAlternatively, use the RL training example which has better fallback:")
        print("   python navisim_isaac/examples/train_rl.py --mode random")
        print("=" * 60)
        return

    # Step 2: Load sector USD scene (if available)
    print("\n[Step 2] Loading sector scene...")
    sector_usd_path = "/Users/jiwon_hae/python_proj/navisim/navisim_isaac/sector_usd.usd"

    if Path(sector_usd_path).exists():
        try:
            scene_prim_path = simulator.load_sector_usd(
                usd_path=sector_usd_path,
                position=(0.0, 0.0, 0.0)
            )
            print(f"✓ Loaded sector USD at: {scene_prim_path}")

            # Load terrain from USD
            terrain_manager = TerrainManager(stage=stage)
            terrain_prim = terrain_manager.load_terrain_from_usd(
                usd_scene_path=scene_prim_path,
                heightmap_prim_name="Heightmap",
                enable_physics=True
            )
            terrain_manager.set_terrain_properties(
                friction=0.8,
                restitution=0.1
            )
            print(f"✓ Loaded terrain with physics: {terrain_prim}")

        except Exception as e:
            print(f"! Could not load sector USD: {e}")
            print("  Continuing with default ground plane...")
    else:
        print(f"! Sector USD not found: {sector_usd_path}")
        print("  Continuing with default ground plane...")

    # Step 3: Spawn navigation robot with camera
    print("\n[Step 3] Spawning navigation robot with camera...")

    # Camera configuration for capturing navigation images
    camera_config = {
        'width': 640,  # Higher resolution for better visualization
        'height': 480,
        'position': (0.3, 0.0, 0.2),  # Mount 30cm forward, 20cm up from robot center
        'orientation': (0.0, 0.0, 0.0, 1.0),  # Looking forward
        'horizontal_aperture': 20.955,
        'focal_length': 24.0
    }

    robot = NavigationRobot(
        robot_type="differential_drive",
        name="nav_robot",
        world=world,
        camera_config=camera_config
    )

    # Spawn robot at initial position
    initial_position = (0.0, 0.0, 0.5)  # Start 0.5m above ground
    robot_prim = robot.spawn(
        position=initial_position,
        orientation=(0.0, 0.0, 0.0, 1.0)  # No rotation
    )
    print(f"✓ Spawned robot at: {initial_position}")
    print(f"  Prim path: {robot_prim}")
    print(f"  Camera enabled: {robot.has_camera()}")

    # Step 4: Initialize navigation controller
    print("\n[Step 4] Initializing navigation controller...")
    controller = NavigationController(
        max_linear_speed=1.0,  # 1 m/s
        max_angular_speed=1.0,  # 1 rad/s
        position_tolerance=0.2,  # 20cm tolerance
        orientation_tolerance=0.1  # ~6 degrees
    )

    # Define waypoint path
    waypoints = [
        (2.0, 0.0, 0.5),   # Move forward 2m
        (2.0, 2.0, 0.5),   # Turn and move right 2m
        (0.0, 2.0, 0.5),   # Move back left 2m
        (0.0, 0.0, 0.5),   # Return to start
    ]
    controller.set_waypoint_path(waypoints)
    print(f"✓ Set navigation path with {len(waypoints)} waypoints")

    # Step 5: Reset world to initialize physics
    print("\n[Step 5] Initializing physics simulation...")
    world.reset()
    print("✓ World reset complete")

    # Step 6: Start simulation with image capture
    print("\n[Step 6] Starting navigation simulation...")
    print("Press Ctrl+C to stop\n")

    # Create output directory for images
    output_dir = Path(__file__).parent / "navigation_output"
    output_dir.mkdir(exist_ok=True)

    # Create timestamped subfolder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = output_dir / f"run_{timestamp}"
    run_output_dir.mkdir(exist_ok=True)
    print(f"✓ Images will be saved to: {run_output_dir}")

    # Debug: Check camera and PIL status
    print(f"✓ Camera enabled: {robot.has_camera()}")
    print(f"✓ PIL available: {PIL_AVAILABLE}")
    print()

    simulator.play()

    # Track saved image count
    images_saved = 0

    try:
        step_count = 0
        max_steps = 3000  # ~50 seconds at 60Hz
        save_interval = 30  # Save image every 30 steps (~0.5 seconds at 60Hz)

        while step_count < max_steps:
            # Get current robot state
            position, orientation = robot.get_pose()

            # Compute velocity commands
            linear_vel, angular_vel = controller.compute_path_following_velocity(
                current_position=position,
                current_orientation=orientation
            )

            # Apply velocity to robot
            robot.set_velocity(linear_vel, angular_vel)

            # Step simulation
            simulator.step(num_steps=1)
            step_count += 1

            # Capture and save camera image periodically
            if step_count % save_interval == 0:
                if not robot.has_camera():
                    if step_count == save_interval:  # Print once
                        print("! Camera not available - skipping image capture")
                elif not PIL_AVAILABLE:
                    if step_count == save_interval:  # Print once
                        print("! PIL not available - skipping image capture")
                else:
                    # Get camera image
                    image = robot.get_camera_image()

                    if image is None:
                        if step_count == save_interval:  # Print once
                            print("! Camera returned None - check camera initialization")
                    else:
                        try:
                            # Save image with step number
                            image_path = run_output_dir / f"step_{step_count:05d}.png"
                            Image.fromarray(image).save(image_path)

                            # Also save a metadata file with robot state
                            metadata_path = run_output_dir / f"step_{step_count:05d}.txt"
                            with open(metadata_path, 'w') as f:
                                f.write(f"Step: {step_count}\n")
                                f.write(f"Position: {position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}\n")
                                f.write(f"Waypoint: {controller.waypoint_index}/{len(waypoints)}\n")
                                f.write(f"Linear Velocity: {linear_vel:.3f} m/s\n")
                                f.write(f"Angular Velocity: {angular_vel:.3f} rad/s\n")

                            images_saved += 1

                            if images_saved == 1:
                                print(f"✓ First image saved to: {image_path.name}")
                        except Exception as e:
                            print(f"! Error saving image at step {step_count}: {e}")

            # Print progress every 60 steps (~1 second)
            if step_count % 60 == 0:
                waypoint_idx = controller.waypoint_index
                total_waypoints = len(waypoints)
                print(f"Step {step_count}: Position {position[:2]}, "
                      f"Waypoint {waypoint_idx}/{total_waypoints}, "
                      f"Vel: ({linear_vel:.2f}, {angular_vel:.2f})")

            # Check if path complete
            if controller.current_waypoint is None:
                print("\n✓ Navigation complete! Path finished.")
                break

    except KeyboardInterrupt:
        print("\n\nNavigation interrupted by user")

    # Step 7: Cleanup
    print("\n[Step 7] Cleaning up...")
    robot.stop()
    simulator.pause()
    print("✓ Simulation paused")

    # Count saved images
    saved_images = list(run_output_dir.glob("*.png"))
    print(f"\n✓ Total images saved: {images_saved}")
    print(f"✓ Images verified in directory: {len(saved_images)}")
    if len(saved_images) > 0:
        print(f"✓ Image directory: {run_output_dir}")
    else:
        print(f"! No images saved - check camera initialization and PIL availability")

    print("\n" + "=" * 60)
    print("Navigation Example Complete!")
    print("=" * 60)
    print(f"\nCamera images saved to: {run_output_dir}")
    print("To close: Stop the simulation and close IsaacSim")


def test_simple_navigation():
    """
    Simplified test for quick validation (without USD scene).
    """
    print("Running Simple Navigation Test...\n")

    # Initialize
    simulator = IsaacSimulator()
    simulator.initialize()
    world = simulator.get_world()

    # Check if IsaacSim is available
    if world is None:
        print("\nERROR: IsaacSim is not available!")
        print("This example requires Isaac Sim. Please install it first.")
        print("See: https://developer.nvidia.com/isaac-sim")
        return

    # Spawn simple robot
    robot = NavigationRobot(name="test_robot", world=world)
    robot.spawn_simple_robot(position=(0.0, 0.0, 0.5), size=(0.5, 0.5, 0.3))

    # Initialize controller
    controller = NavigationController()
    controller.set_waypoint((2.0, 2.0, 0.5))

    # Reset and play
    world.reset()
    simulator.play()

    # Run for 300 steps
    for i in range(300):
        pos, orn = robot.get_pose()
        lin_vel, ang_vel = controller.compute_velocity_to_point(pos, orn, (2.0, 2.0, 0.5))
        robot.set_velocity(lin_vel, ang_vel)
        simulator.step()

        if i % 60 == 0:
            print(f"Step {i}: Position {pos}, Velocity ({lin_vel:.2f}, {ang_vel:.2f})")

    robot.stop()
    simulator.pause()
    print("\nSimple test complete!")


if __name__ == "__main__":
    import sys

    if "--simple" in sys.argv:
        test_simple_navigation()
    else:
        main()
