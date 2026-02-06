#!/usr/bin/env python3
"""Fix Jetbot camera transform order to match Isaac Lab's expectations.

This script converts rotateZYX to orient (quaternion) in the camera prim.
Run this BEFORE launching your training/play scripts.

Usage:
    python scripts/fix_jetbot_camera_transforms.py --usd_path path/to/jetbot.usd
"""

import argparse
from pathlib import Path


def fix_camera_transforms(usd_path: Path):
    """Fix camera transform order in USD file.

    Converts xformOp:rotateZYX to xformOp:orient (quaternion).
    """
    try:
        from pxr import Usd, UsdGeom, Gf
        import numpy as np
    except ImportError:
        print("Error: pxr (USD) not found. Install with: pip install usd-core")
        return False

    print(f"Opening USD: {usd_path}")

    # Open the USD stage
    stage = Usd.Stage.Open(str(usd_path))

    # Find camera prim
    camera_path = "/Jetbot/chassis/rgb_camera/jetbot_camera"
    camera_prim = stage.GetPrimAtPath(camera_path)

    if not camera_prim.IsValid():
        print(f"Error: Camera prim not found at {camera_path}")
        # Try to find it
        print("Searching for camera prims...")
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                print(f"  Found camera: {prim.GetPath()}")
        return False

    print(f"Found camera prim: {camera_path}")

    # Get xformable
    xformable = UsdGeom.Xformable(camera_prim)

    # Get current transform ops
    xform_ops = xformable.GetOrderedXformOps()

    print(f"Current transform ops: {[op.GetOpName() for op in xform_ops]}")

    # Check if rotateZYX exists
    has_rotate_zyx = any(op.GetOpName() == "xformOp:rotateZYX" for op in xform_ops)
    has_orient = any(op.GetOpName() == "xformOp:orient" for op in xform_ops)

    if not has_rotate_zyx:
        print("No rotateZYX found - camera already in correct format or different rotation type")
        return True

    if has_orient:
        print("Camera already has orient - skipping")
        return True

    print("Converting rotateZYX to orient...")

    # Get the rotation values
    rotate_op = None
    for op in xform_ops:
        if op.GetOpName() == "xformOp:rotateZYX":
            rotate_op = op
            break

    if rotate_op:
        # Get Euler angles (in degrees)
        euler_angles = rotate_op.Get()  # (Z, Y, X) in degrees
        print(f"  Current rotation (ZYX): {euler_angles}")

        # Convert to radians
        euler_rad = [np.radians(angle) for angle in euler_angles]

        # Convert ZYX Euler to quaternion
        # Rotation order: Z (yaw), then Y (pitch), then X (roll)
        z, y, x = euler_rad

        # Create rotation matrices
        cos_x, sin_x = np.cos(x/2), np.sin(x/2)
        cos_y, sin_y = np.cos(y/2), np.sin(y/2)
        cos_z, sin_z = np.cos(z/2), np.sin(z/2)

        # Quaternion multiplication (ZYX order)
        qw = cos_z * cos_y * cos_x + sin_z * sin_y * sin_x
        qx = cos_z * cos_y * sin_x - sin_z * sin_y * cos_x
        qy = cos_z * sin_y * cos_x + sin_z * cos_y * sin_x
        qz = sin_z * cos_y * cos_x - cos_z * sin_y * sin_x

        quat = Gf.Quatf(qw, qx, qy, qz)
        print(f"  Converted to quaternion: w={qw:.4f}, x={qx:.4f}, y={qy:.4f}, z={qz:.4f}")

        # Clear existing transform ops
        xformable.ClearXformOpOrder()

        # Re-add in canonical order: translate, orient, scale
        translate_op = xformable.AddTranslateOp()
        orient_op = xformable.AddOrientOp()
        scale_op = xformable.AddScaleOp()

        # Set values (get from original ops if they exist)
        for op in xform_ops:
            if "translate" in op.GetOpName():
                translate_op.Set(op.Get())
            elif "scale" in op.GetOpName():
                scale_op.Set(op.Get())

        # Set the new quaternion
        orient_op.Set(quat)

        print("✓ Transform order fixed!")
        print(f"  New order: {[op.GetOpName() for op in xformable.GetOrderedXformOps()]}")

        # Save the stage
        stage.Save()
        print(f"✓ Saved USD file: {usd_path}")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Fix Jetbot camera transform order for Isaac Lab compatibility"
    )
    parser.add_argument(
        "--usd_path",
        type=Path,
        required=True,
        help="Path to Jetbot USD file"
    )

    args = parser.parse_args()

    if not args.usd_path.exists():
        print(f"Error: USD file not found: {args.usd_path}")
        return 1

    success = fix_camera_transforms(args.usd_path)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
