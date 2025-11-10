#!/usr/bin/env python3
"""
Test Camera Integration with IsaacSim

This script demonstrates the proper initialization sequence and camera usage.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from navisim_isaac.isaac.simulator import IsaacSimulator
from navisim_isaac.isaac.navigation_robot import NavigationRobot


def test_camera_integration():
    """
    Test that camera is properly initialized and captures images.
    """
    print("=" * 70)
    print("Testing Camera Integration")
    print("=" * 70)

    # Configuration
    config = {
        'physics_dt': 1.0 / 60.0,
        'rendering_dt': 1.0 / 30.0,
    }

    # Initialize IsaacSim (this creates SimulationApp first)
    print("\n[1/4] Initializing IsaacSim...")
    simulator = IsaacSimulator(config=config)
    simulator.initialize(headless=True)  # Headless for faster execution

    world = simulator.get_world()

    if world is None:
        print("\nERROR: IsaacSim is not available!")
        print("Please install Isaac Sim and run with Isaac Sim's Python:")
        print("  ~/.local/share/ov/pkg/isaac_sim-*/python.sh navisim_isaac/examples/test_camera.py")
        return

    print("✓ IsaacSim initialized")

    # Create robot with camera
    print("\n[2/4] Creating robot with camera...")
    camera_config = {
        'width': 256,
        'height': 256,
        'position': (0.3, 0.0, 0.2),
        'orientation': (0.0, 0.0, 0.0, 1.0),
        'horizontal_aperture': 20.955,
        'focal_length': 24.0
    }

    robot = NavigationRobot(
        robot_type='differential_drive',
        name='test_robot',
        world=world,
        camera_config=camera_config
    )

    robot.spawn(position=(0.0, 0.0, 0.5))
    print(f"✓ Robot spawned with camera: {robot.has_camera()}")

    # Reset and start simulation
    print("\n[3/4] Starting simulation...")
    world.reset()
    simulator.play()
    print("✓ Simulation started")

    # Test camera capture
    print("\n[4/4] Testing camera capture...")

    # Take a few steps and capture images
    for step in range(10):
        # Move robot
        robot.set_velocity(0.1, 0.0)
        simulator.step(num_steps=1)

        # Capture image
        image = robot.get_camera_image()

        if image is not None:
            print(f"  Step {step}: Captured image shape: {image.shape}, dtype: {image.dtype}")

            # Check if image has content (not all zeros)
            has_content = np.any(image > 0)
            print(f"    Image has content: {has_content}")
        else:
            print(f"  Step {step}: No image captured")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

    # Cleanup
    robot.stop()
    simulator.shutdown()


if __name__ == "__main__":
    try:
        test_camera_integration()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
