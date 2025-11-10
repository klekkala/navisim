#!/usr/bin/env python3
"""
Camera Debug Test

Quick diagnostic to check camera initialization and image capture.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from navisim_isaac.isaac.simulator import IsaacSimulator
from navisim_isaac.isaac.navigation_robot import NavigationRobot

# Check PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
    print("✓ PIL (Pillow) is available")
except ImportError:
    PIL_AVAILABLE = False
    print("✗ PIL not available - install with: pip install Pillow")

print("\n" + "=" * 70)
print("Camera Debug Test")
print("=" * 70)

# Initialize IsaacSim
print("\n[1/5] Initializing IsaacSim...")
simulator = IsaacSimulator()
simulator.initialize(headless=True)

world = simulator.get_world()

if world is None:
    print("\n✗ IsaacSim is not available!")
    print("This test requires Isaac Sim. Run with:")
    print("  ~/.local/share/ov/pkg/isaac_sim-*/python.sh navisim_isaac/examples/test_camera_debug.py")
    sys.exit(1)

print("✓ IsaacSim initialized")

# Create robot with camera
print("\n[2/5] Creating robot with camera...")
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

print("✓ Robot created")

# Spawn robot
print("\n[3/5] Spawning robot...")
robot.spawn(position=(0.0, 0.0, 0.5))
print(f"✓ Robot spawned")

# Check camera status
print("\n[4/5] Checking camera status...")
print(f"  has_camera(): {robot.has_camera()}")
print(f"  camera object: {robot.camera}")

if not robot.has_camera():
    print("\n✗ Camera NOT initialized!")
    print("Check the error messages above for details.")
    simulator.shutdown()
    sys.exit(1)

print("✓ Camera is initialized")

# Reset and start simulation
print("\n[5/5] Testing image capture...")
world.reset()
simulator.play()

# Take a few steps and try to capture images
success_count = 0
for i in range(5):
    # Step simulation
    simulator.step(num_steps=1)

    # Try to capture image
    image = robot.get_camera_image()

    if image is not None:
        print(f"  Step {i}: Captured {image.shape}, dtype={image.dtype}, "
              f"min={image.min()}, max={image.max()}")
        success_count += 1

        # Save first image if PIL available
        if i == 0 and PIL_AVAILABLE:
            output_dir = Path(__file__).parent / "navigation_output"
            output_dir.mkdir(exist_ok=True)
            test_image_path = output_dir / "test_camera_capture.png"
            Image.fromarray(image).save(test_image_path)
            print(f"  ✓ Saved test image to: {test_image_path}")
    else:
        print(f"  Step {i}: Image is None!")

print("\n" + "=" * 70)
if success_count > 0:
    print(f"✓ SUCCESS: Captured {success_count}/5 images")
    print("=" * 70)
    print("\nCamera is working! You can now run navigation_example.py")
else:
    print("✗ FAILED: No images captured")
    print("=" * 70)
    print("\nCheck error messages above for debugging information")

# Cleanup
robot.stop()
simulator.shutdown()
