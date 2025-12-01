# scripts/inspect_jetbot_camera.py

"""Inspect Jetbot USD to find camera configuration."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys
import os
from pxr import Usd, UsdGeom

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.paths import JETBOT_USD
import omni.usd


def inspect_usd_cameras(usd_path: str):
    """Inspect USD file for camera prims."""
    print(f"\n{'='*80}")
    print(f"Inspecting USD: {usd_path}")
    print(f"{'='*80}\n")

    # Load USD stage
    stage = Usd.Stage.Open(usd_path)

    if not stage:
        print("ERROR: Could not load USD file")
        return

    # Find all camera prims
    print("Searching for cameras in USD hierarchy...\n")

    cameras_found = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            cameras_found.append(prim.GetPath().pathString)
            print(f"✓ Found Camera: {prim.GetPath().pathString}")

            # Get camera properties
            camera = UsdGeom.Camera(prim)
            focal_length = camera.GetFocalLengthAttr().Get()
            horiz_aperture = camera.GetHorizontalApertureAttr().Get()
            vert_aperture = camera.GetVerticalApertureAttr().Get()

            print(f"  - Focal Length: {focal_length}")
            print(f"  - Horizontal Aperture: {horiz_aperture}")
            print(f"  - Vertical Aperture: {vert_aperture}")
            print()

    if not cameras_found:
        print("No cameras found in USD file.")
        print("\nListing all prims to help locate camera:")
        for prim in stage.Traverse():
            print(f"  {prim.GetPath().pathString} [{prim.GetTypeName()}]")
    else:
        print(f"\n{'='*80}")
        print(f"Summary: Found {len(cameras_found)} camera(s)")
        print(f"{'='*80}")
        for cam_path in cameras_found:
            print(f"  - {cam_path}")

    simulation_app.close()


if __name__ == "__main__":
    inspect_usd_cameras(JETBOT_USD)
