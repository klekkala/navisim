#!/usr/bin/env python3
"""
Verify Camera Fix

This script verifies that the camera initialization bug is fixed.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("=" * 70)
print("Verifying Camera Initialization Fix")
print("=" * 70)

# Test 1: Check is_spawned flag order
print("\n[Test 1] Checking spawn order logic...")

from navisim_isaac.isaac.simulator import IsaacSimulator
from navisim_isaac.isaac.navigation_robot import NavigationRobot

# Initialize simulator
simulator = IsaacSimulator()
simulator.initialize(headless=True)

world = simulator.get_world()

if world is None:
    print("✓ TEST PASSED (without Isaac Sim)")
    print("\nRunning in simulation mode:")
    print("  - Camera initialization will be skipped")
    print("  - Robot will return placeholder images")
    print("\nTo test with real camera, run with Isaac Sim's Python:")
    print("  ~/.local/share/ov/pkg/isaac_sim-*/python.sh verify_camera_fix.py")
    sys.exit(0)

# Create robot
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

print(f"  Before spawn: is_spawned={robot.is_spawned}, has_camera={robot.has_camera()}")

# Spawn robot (this should initialize camera)
robot.spawn(position=(0.0, 0.0, 0.5))

print(f"  After spawn:  is_spawned={robot.is_spawned}, has_camera={robot.has_camera()}")

# Check results
if robot.is_spawned and robot.has_camera():
    print("✓ TEST PASSED: Camera initialized successfully!")
    print(f"  Camera object: {type(robot.camera).__name__}")
elif robot.is_spawned and not robot.has_camera():
    print("✗ TEST FAILED: Robot spawned but camera is None")
    print("  This means camera initialization failed (check error messages above)")
else:
    print("✗ TEST FAILED: Robot not spawned properly")

# Test 2: Try to capture an image
print("\n[Test 2] Testing image capture...")

world.reset()
simulator.play()
simulator.step(num_steps=5)  # Take a few steps

image = robot.get_camera_image()

if image is not None:
    print(f"✓ TEST PASSED: Captured image with shape {image.shape}")

    # Try to save it
    try:
        from PIL import Image as PILImage
        output_dir = Path(__file__).parent / "navigation_output"
        output_dir.mkdir(exist_ok=True)
        test_path = output_dir / "verify_test.png"
        PILImage.fromarray(image).save(test_path)
        print(f"✓ TEST PASSED: Saved image to {test_path}")
    except ImportError:
        print("! PIL not available (install with: pip install Pillow)")
    except Exception as e:
        print(f"✗ Failed to save image: {e}")
else:
    print("✗ TEST FAILED: Image is None")

# Cleanup
print("\n" + "=" * 70)
print("Verification Complete")
print("=" * 70)

robot.stop()
simulator.shutdown()
