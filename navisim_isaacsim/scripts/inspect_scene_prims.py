# scripts/inspect_scene_prims.py

"""Inspect what prims exist in the NaviSim scene after environment setup."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.navigation_env_cfg import NavisimNavigationEnvCfg
from tasks.navigation_env import NavisimNavigationEnv
from pxr import Usd, UsdGeom


def inspect_scene_hierarchy(stage, max_depth=10):
    """Print the USD scene hierarchy."""
    print(f"\n{'='*80}")
    print("USD Scene Hierarchy")
    print(f"{'='*80}\n")

    def print_prim(prim, depth=0):
        if depth > max_depth:
            return

        indent = "  " * depth
        prim_type = prim.GetTypeName()
        prim_path = prim.GetPath().pathString

        # Highlight cameras and robots
        marker = ""
        if prim.IsA(UsdGeom.Camera):
            marker = " [CAMERA]"
        elif "Jetbot" in prim_path or "jetbot" in prim_path:
            marker = " [JETBOT]"
        elif "warehouse" in prim_path.lower():
            marker = " [WAREHOUSE]"

        print(f"{indent}{prim.GetName()} ({prim_type}){marker}")

        # Recurse to children
        for child in prim.GetChildren():
            print_prim(child, depth + 1)

    # Start from root
    root = stage.GetPseudoRoot()
    for prim in root.GetChildren():
        print_prim(prim)


def find_cameras(stage):
    """Find all camera prims in the stage."""
    print(f"\n{'='*80}")
    print("Camera Prims Found")
    print(f"{'='*80}\n")

    cameras = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            cameras.append(prim.GetPath().pathString)
            print(f"✓ {prim.GetPath().pathString}")

    if not cameras:
        print("No cameras found in scene!")

    return cameras


def find_jetbot_prims(stage):
    """Find all Jetbot-related prims."""
    print(f"\n{'='*80}")
    print("Jetbot-Related Prims")
    print(f"{'='*80}\n")

    jetbot_prims = []
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        if "Jetbot" in prim_path or "jetbot" in prim_path:
            jetbot_prims.append(prim_path)
            print(f"  {prim_path} [{prim.GetTypeName()}]")

    if not jetbot_prims:
        print("No Jetbot prims found!")

    return jetbot_prims


def main():
    """Inspect scene after environment creation."""

    # Create environment
    env_cfg = NavisimNavigationEnvCfg()
    env_cfg.scene.num_envs = 1

    print(f"\n{'='*80}")
    print("Creating NaviSim Environment")
    print(f"{'='*80}\n")

    env = NavisimNavigationEnv(
        cfg=env_cfg,
        render_mode=None  # Headless
    )

    print("Environment created successfully!")

    # Reset to spawn everything
    obs, _ = env.reset()
    print("Environment reset complete\n")

    # Get USD stage
    stage = env.scene.stage

    # Inspect scene
    inspect_scene_hierarchy(stage, max_depth=5)

    # Find specific elements
    cameras = find_cameras(stage)
    jetbot_prims = find_jetbot_prims(stage)

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"  Cameras found: {len(cameras)}")
    print(f"  Jetbot prims found: {len(jetbot_prims)}")
    print(f"{'='*80}\n")

    if cameras:
        print("Camera paths to try:")
        for cam in cameras:
            print(f"  {cam}")
    else:
        print("WARNING: No cameras found in scene!")
        print("The Jetbot USD may not include a camera, or it wasn't loaded properly.")

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
