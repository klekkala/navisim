"""Utility to build SceneGraph from section data.

This script helps convert your existing section data and NetworkX graph
into the format needed by DynamicSceneManager.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import networkx as nx

from .scene_graph import SceneGraph, SceneSection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_scene_graph_from_data(
    sections_data: List[Dict],
    graph: Optional[nx.Graph] = None,
    auto_connect_nearby: bool = True,
    connection_threshold: float = 60.0,
) -> SceneGraph:
    """Build a SceneGraph from section data.

    Args:
        sections_data: List of section dictionaries with keys:
            - 'id': Section identifier
            - 'usd_path': Path to USD file
            - 'bounds': [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            - 'center': [x, y, z]
            - 'metadata': Optional dict with extra info
            - 'neighbors': Optional list of neighbor section IDs
        graph: Optional pre-built NetworkX graph
        auto_connect_nearby: Automatically connect nearby sections
        connection_threshold: Distance threshold for auto-connection (meters)

    Returns:
        SceneGraph instance

    Example:
        sections_data = [
            {
                'id': 'section_001',
                'usd_path': 'assets/sections/section_001.usd',
                'bounds': [[0, 0, 0], [50, 50, 10]],
                'center': [25, 25, 5],
                'metadata': {'type': 'warehouse', 'difficulty': 'easy'},
                'neighbors': ['section_002']
            },
            ...
        ]
        scene_graph = build_scene_graph_from_data(sections_data)
        scene_graph.to_pickle('scene_graph.pkl')
    """
    scene_graph = SceneGraph(graph)

    logger.info(f"Building scene graph from {len(sections_data)} sections")

    # Add all sections first
    for data in sections_data:
        scene_graph.add_section(
            section_id=data["id"],
            usd_path=Path(data["usd_path"]),
            bounds=np.array(data["bounds"]),
            center=np.array(data["center"]),
            metadata=data.get("metadata"),
            neighbors=data.get("neighbors"),
        )

    # Auto-connect nearby sections if requested
    if auto_connect_nearby:
        logger.info(
            f"Auto-connecting sections within {connection_threshold}m"
        )

        section_ids = list(scene_graph.sections.keys())
        connections_added = 0

        for i, id1 in enumerate(section_ids):
            for id2 in section_ids[i + 1 :]:
                section1 = scene_graph.sections[id1]
                section2 = scene_graph.sections[id2]

                dist = np.linalg.norm(section1.center - section2.center)

                if dist <= connection_threshold:
                    # Check if not already connected
                    if not scene_graph.graph.has_edge(id1, id2):
                        scene_graph.graph.add_edge(id1, id2)
                        connections_added += 1

        logger.info(f"Added {connections_added} automatic connections")

    logger.info(
        f"Scene graph built: {len(scene_graph)} sections, "
        f"{scene_graph.graph.number_of_edges()} connections"
    )

    return scene_graph


def convert_navisim_sequence_graph(
    sequence_graph_path: Path,
    sections_root: Path,
    output_path: Path,
) -> SceneGraph:
    """Convert NaviSim sequence graph to SceneGraph format.

    This assumes you have a NaviSim-style sequence graph with sectors
    that have been converted to USD files.

    Args:
        sequence_graph_path: Path to NaviSim sequence_graph.gpickle
        sections_root: Root directory containing USD section files
        output_path: Where to save the converted scene graph

    Returns:
        Converted SceneGraph
    """
    import pickle

    logger.info(f"Loading NaviSim sequence graph from: {sequence_graph_path}")

    with open(sequence_graph_path, "rb") as f:
        navisim_graph = pickle.load(f)

    sections_data = []

    # Extract section data from NaviSim graph
    # Note: Adjust this based on your actual data structure
    for node_id, node_data in navisim_graph.nodes(data=True):
        # Assuming your sections are stored as node attributes
        # Adjust keys based on your actual format

        usd_path = sections_root / f"{node_id}.usd"

        if not usd_path.exists():
            logger.warning(f"USD file not found for section {node_id}: {usd_path}")
            continue

        # Extract or compute bounds and center
        # You may need to adjust these based on your data format
        bounds = node_data.get(
            "bounds", [[0, 0, 0], [50, 50, 10]]
        )  # Default if not available
        center = node_data.get("center", [25, 25, 5])

        # Get neighbors from graph
        neighbors = list(navisim_graph.neighbors(node_id))

        sections_data.append(
            {
                "id": node_id,
                "usd_path": str(usd_path),
                "bounds": bounds,
                "center": center,
                "metadata": node_data.get("metadata", {}),
                "neighbors": neighbors,
            }
        )

    # Build scene graph
    scene_graph = build_scene_graph_from_data(
        sections_data, graph=navisim_graph, auto_connect_nearby=False
    )

    # Save converted graph
    scene_graph.to_pickle(output_path)
    logger.info(f"Saved converted scene graph to: {output_path}")

    return scene_graph


def main():
    """Command-line interface for building scene graphs."""
    parser = argparse.ArgumentParser(description="Build scene graph from section data")
    parser.add_argument(
        "--navisim_graph",
        type=Path,
        help="Path to NaviSim sequence_graph.gpickle",
    )
    parser.add_argument(
        "--sections_root",
        type=Path,
        help="Root directory containing USD section files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for scene graph pickle",
    )
    parser.add_argument(
        "--visualize",
        type=Path,
        help="Optional: save visualization to this path",
    )

    args = parser.parse_args()

    if args.navisim_graph:
        # Convert from NaviSim format
        scene_graph = convert_navisim_sequence_graph(
            args.navisim_graph, args.sections_root, args.output
        )
    else:
        # TODO: Add support for other input formats
        logger.error("Must specify --navisim_graph for conversion")
        return

    logger.info(f"Scene graph created: {scene_graph}")

    if args.visualize:
        scene_graph.visualize(args.visualize)


if __name__ == "__main__":
    main()
