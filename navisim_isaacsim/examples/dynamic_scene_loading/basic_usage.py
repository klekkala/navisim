#!/usr/bin/env python3
"""Basic example of dynamic scene loading in NaviSim.

This example demonstrates:
1. Building a scene graph from section data
2. Wrapping an environment with dynamic loading
3. Observing section transitions during navigation
"""

import argparse
import numpy as np
from pathlib import Path

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Dynamic scene loading example")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--scene_graph", type=Path, help="Path to scene_graph.pkl")
parser.add_argument("--build_demo", action="store_true", help="Build demo scene graph")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after AppLauncher
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import navisim_lab.tasks

from navisim_lab.scene import SceneGraph, DynamicSceneEnvWrapper
from navisim_lab.scene.build_scene_graph import build_scene_graph_from_data


def build_demo_scene_graph() -> SceneGraph:
    """Build a simple demo scene graph for testing."""
    print("Building demo scene graph...")

    # Create a 3x3 grid of sections
    sections_data = []

    for i in range(3):
        for j in range(3):
            section_id = f"section_{i}_{j}"

            # Each section is 50m x 50m
            min_x, min_y = i * 50, j * 50
            max_x, max_y = (i + 1) * 50, (j + 1) * 50
            center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

            # For demo, USD files don't need to exist
            # In real use, replace with actual paths
            usd_path = f"assets/sections/{section_id}.usd"

            # Determine neighbors (up, down, left, right)
            neighbors = []
            if i > 0:
                neighbors.append(f"section_{i-1}_{j}")
            if i < 2:
                neighbors.append(f"section_{i+1}_{j}")
            if j > 0:
                neighbors.append(f"section_{i}_{j-1}")
            if j < 2:
                neighbors.append(f"section_{i}_{j+1}")

            sections_data.append(
                {
                    "id": section_id,
                    "usd_path": usd_path,
                    "bounds": [[min_x, min_y, 0], [max_x, max_y, 10]],
                    "center": [center_x, center_y, 5],
                    "metadata": {"grid_pos": (i, j)},
                    "neighbors": neighbors,
                }
            )

    scene_graph = build_scene_graph_from_data(sections_data, auto_connect_nearby=False)

    print(f"Created demo scene graph: {len(scene_graph)} sections")
    return scene_graph


def main():
    print("=" * 80)
    print("NaviSim Dynamic Scene Loading - Basic Example")
    print("=" * 80)

    # Build or load scene graph
    if args.build_demo:
        scene_graph = build_demo_scene_graph()

        # Save for future use
        output_path = Path("demo_scene_graph.pkl")
        scene_graph.to_pickle(output_path)
        print(f"Saved demo scene graph to: {output_path}")

        # Visualize
        try:
            scene_graph.visualize("demo_scene_graph.png")
            print("Saved visualization to: demo_scene_graph.png")
        except ImportError:
            print("matplotlib not available, skipping visualization")

    elif args.scene_graph:
        print(f"Loading scene graph from: {args.scene_graph}")
        scene_graph = SceneGraph.from_pickle(args.scene_graph)
        print(f"Loaded scene graph: {len(scene_graph)} sections")
    else:
        print("Error: Must specify --scene_graph or --build_demo")
        return

    # Create environment
    print("\nCreating environment...")
    env_cfg = parse_env_cfg("Navisim-Warehouse-Jetbot-v0", num_envs=args.num_envs)
    env = gym.make("Navisim-Warehouse-Jetbot-v0", cfg=env_cfg)
    print("Environment created")

    # Wrap with dynamic scene loading
    print("\nWrapping with dynamic scene loading...")
    env = DynamicSceneEnvWrapper(
        env,
        scene_graph=scene_graph,
        max_loaded_sections=9,
        load_radius=50.0,
        neighbor_depth=1,
        update_frequency=10,
        preload_sections=[
            list(scene_graph.sections.keys())[0]
        ],  # Preload first section
    )
    print("Dynamic scene loading enabled")

    # Run simulation
    print("\n" + "=" * 80)
    print("Running simulation...")
    print("=" * 80)

    obs, info = env.reset()

    # Print initial scene state
    print(f"\nInitial section: {info.get('current_section')}")
    print(f"Scene stats: {info.get('scene_stats')}")

    num_steps = 1000
    section_transitions = 0

    for step in range(num_steps):
        # Simple forward action
        action = env.sample_forward_action()
        obs, reward, terminated, truncated, info = env.step(action)

        # Log scene updates
        if "scene_update" in info:
            update = info["scene_update"]
            if update["loaded"] or update["unloaded"]:
                print(
                    f"\n[Step {step}] Scene Update:"
                )
                if update["loaded"]:
                    print(f"  Loaded:   {update['loaded']}")
                if update["unloaded"]:
                    print(f"  Unloaded: {update['unloaded']}")

                stats = info.get("scene_stats", {})
                print(
                    f"  Currently loaded: {stats.get('loaded_sections')}/{stats.get('total_sections')}"
                )

        # Log section transitions
        if "section_transition" in info:
            transition = info["section_transition"]
            section_transitions += 1
            print(
                f"\n[Step {step}] Section Transition:"
            )
            print(f"  From: {transition['from']}")
            print(f"  To:   {transition['to']}")

        # Progress indicator
        if (step + 1) % 100 == 0:
            print(f"\nProgress: {step+1}/{num_steps} steps completed")
            stats = env.get_scene_stats()
            print(f"  Loaded sections: {stats['loaded_sections']}")
            print(f"  Total loads: {stats['total_loads']}")
            print(f"  Total unloads: {stats['total_unloads']}")

        if terminated.any() or truncated.any():
            print("\nEpisode ended, resetting...")
            obs, info = env.reset()

    # Final statistics
    print("\n" + "=" * 80)
    print("Simulation Complete")
    print("=" * 80)

    final_stats = env.get_scene_stats()
    print(f"\nFinal Statistics:")
    print(f"  Total sections: {final_stats['total_sections']}")
    print(f"  Total loads: {final_stats['total_loads']}")
    print(f"  Total unloads: {final_stats['total_unloads']}")
    print(f"  Section transitions: {section_transitions}")
    print(f"  Average sections loaded: {final_stats['loaded_sections']}")

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
