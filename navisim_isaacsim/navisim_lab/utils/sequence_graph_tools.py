"""Utility tools for working with sequence graphs and RocksDB.

Provides functions to convert NaviSim sequence graphs to Isaac Lab format,
populate RocksDB with USD metadata, and manage section data.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from ..scene import SequenceGraph, SceneGraph
from ..data import get_db

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_rocksdb_from_sequence_graph(
    sequence_graph_path: Path,
    usd_root: Path,
    default_bounds: Optional[List] = None,
    overwrite: bool = False,
):
    """Set up RocksDB with USD metadata from a NaviSim sequence graph.

    This function:
    1. Loads the sequence graph
    2. Scans for USD files in usd_root
    3. Populates RocksDB with section metadata

    Args:
        sequence_graph_path: Path to sequence_graph.gpickle
        usd_root: Root directory containing USD files
        default_bounds: Default bounds [[min], [max]] if not computed
        overwrite: Whether to overwrite existing entries

    Example:
        >>> setup_rocksdb_from_sequence_graph(
        ...     Path("assets/sequence_graph.gpickle"),
        ...     Path("assets/usd_sections"),
        ...     default_bounds=[[0, 0, 0], [50, 50, 10]]
        ... )
    """
    logger.info("Setting up RocksDB from sequence graph...")
    logger.info(f"Sequence graph: {sequence_graph_path}")
    logger.info(f"USD root: {usd_root}")

    # Load sequence graph
    seq_graph = SequenceGraph(sequence_graph_path, use_rocksdb=True)

    # Populate RocksDB
    seq_graph.populate_rocksdb_from_directory(
        usd_root=usd_root,
        default_bounds=default_bounds,
    )

    logger.info("RocksDB setup complete!")


def convert_sequence_graph_to_scene_graph(
    sequence_graph_path: Path,
    output_path: Path,
    usd_root: Optional[Path] = None,
    default_bounds: Optional[List] = None,
):
    """Convert NaviSim sequence graph to SceneGraph format and save.

    Args:
        sequence_graph_path: Path to sequence_graph.gpickle
        output_path: Where to save the SceneGraph pickle
        usd_root: Root directory for USD files (fallback if no RocksDB)
        default_bounds: Default bounds if not in metadata

    Example:
        >>> convert_sequence_graph_to_scene_graph(
        ...     Path("assets/sequence_graph.gpickle"),
        ...     Path("assets/scene_graph.pkl")
        ... )
    """
    logger.info("Converting sequence graph to scene graph...")

    # Load sequence graph (will use RocksDB if available)
    seq_graph = SequenceGraph(sequence_graph_path, use_rocksdb=True)

    # Convert to scene graph
    scene_graph = seq_graph.to_scene_graph(
        usd_root=usd_root,
        default_bounds=default_bounds,
    )

    # Save to pickle
    scene_graph.to_pickle(output_path)

    logger.info(f"Scene graph saved to: {output_path}")
    logger.info(f"Sections: {len(scene_graph)}")
    logger.info(f"Edges: {scene_graph.graph.number_of_edges()}")


def visualize_sequence_graph(
    sequence_graph_path: Path,
    output_image: Path,
):
    """Visualize a sequence graph.

    Args:
        sequence_graph_path: Path to sequence_graph.gpickle
        output_image: Where to save the visualization
    """
    logger.info(f"Visualizing sequence graph from {sequence_graph_path}...")

    seq_graph = SequenceGraph(sequence_graph_path, use_rocksdb=False)

    # Convert to scene graph for visualization
    scene_graph = seq_graph.to_scene_graph()
    scene_graph.visualize(output_image)

    logger.info(f"Visualization saved to: {output_image}")


def list_sections_in_rocksdb():
    """List all sections stored in RocksDB."""
    db = get_db()

    sections = db.list_sections()

    logger.info(f"Found {len(sections)} sections in RocksDB:")

    for section_id in sections:
        metadata = db.get_section_metadata(section_id)
        if metadata:
            usd_path = metadata.get("usd_path", "N/A")
            logger.info(f"  - {section_id}: {usd_path}")
        else:
            logger.info(f"  - {section_id}: (no metadata)")


def verify_usd_files(
    sequence_graph_path: Path,
    usd_root: Path,
) -> Dict[str, bool]:
    """Verify that USD files exist for all sections in the sequence graph.

    Args:
        sequence_graph_path: Path to sequence_graph.gpickle
        usd_root: Root directory to search for USD files

    Returns:
        Dict mapping section_id to whether USD file exists
    """
    logger.info("Verifying USD files...")

    seq_graph = SequenceGraph(sequence_graph_path, use_rocksdb=False)

    results = {}
    missing = []

    for seq_id, sector_ids in seq_graph.sequences.items():
        for sector_id in sector_ids:
            section_id = f"{seq_id}/{sector_id}"

            # Try multiple naming conventions
            possible_paths = [
                usd_root / f"{seq_id}_{sector_id}.usd",
                usd_root / seq_id / f"{sector_id}.usd",
                usd_root / seq_id / f"{sector_id}" / "scene.usd",
            ]

            found = False
            for path in possible_paths:
                if path.exists():
                    results[section_id] = True
                    found = True
                    break

            if not found:
                results[section_id] = False
                missing.append(section_id)

    logger.info(f"Verification complete:")
    logger.info(f"  Total sections: {len(results)}")
    logger.info(f"  Found USD files: {sum(results.values())}")
    logger.info(f"  Missing USD files: {len(missing)}")

    if missing:
        logger.warning("Missing USD files for:")
        for section_id in missing[:10]:  # Show first 10
            logger.warning(f"  - {section_id}")
        if len(missing) > 10:
            logger.warning(f"  ... and {len(missing) - 10} more")

    return results


def main():
    """Command-line interface for sequence graph tools."""
    parser = argparse.ArgumentParser(
        description="Utilities for NaviSim sequence graphs and RocksDB"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Setup RocksDB
    setup_parser = subparsers.add_parser(
        "setup",
        help="Set up RocksDB from sequence graph and USD files"
    )
    setup_parser.add_argument(
        "--sequence_graph",
        type=Path,
        required=True,
        help="Path to sequence_graph.gpickle",
    )
    setup_parser.add_argument(
        "--usd_root",
        type=Path,
        required=True,
        help="Root directory containing USD files",
    )
    setup_parser.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        help="Default bounds: min_x min_y min_z max_x max_y max_z",
    )

    # Convert to scene graph
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert sequence graph to scene graph format"
    )
    convert_parser.add_argument(
        "--sequence_graph",
        type=Path,
        required=True,
        help="Path to sequence_graph.gpickle",
    )
    convert_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for scene graph pickle",
    )
    convert_parser.add_argument(
        "--usd_root",
        type=Path,
        help="Root directory for USD files (fallback)",
    )

    # Visualize
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Visualize sequence graph"
    )
    viz_parser.add_argument(
        "--sequence_graph",
        type=Path,
        required=True,
        help="Path to sequence_graph.gpickle",
    )
    viz_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output image path",
    )

    # List sections
    list_parser = subparsers.add_parser(
        "list",
        help="List sections in RocksDB"
    )

    # Verify USD files
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify USD files exist"
    )
    verify_parser.add_argument(
        "--sequence_graph",
        type=Path,
        required=True,
        help="Path to sequence_graph.gpickle",
    )
    verify_parser.add_argument(
        "--usd_root",
        type=Path,
        required=True,
        help="Root directory to search for USD files",
    )

    args = parser.parse_args()

    if args.command == "setup":
        bounds = None
        if args.bounds:
            bounds = [
                [args.bounds[0], args.bounds[1], args.bounds[2]],
                [args.bounds[3], args.bounds[4], args.bounds[5]],
            ]

        setup_rocksdb_from_sequence_graph(
            args.sequence_graph,
            args.usd_root,
            default_bounds=bounds,
        )

    elif args.command == "convert":
        convert_sequence_graph_to_scene_graph(
            args.sequence_graph,
            args.output,
            usd_root=args.usd_root,
        )

    elif args.command == "visualize":
        visualize_sequence_graph(
            args.sequence_graph,
            args.output,
        )

    elif args.command == "list":
        list_sections_in_rocksdb()

    elif args.command == "verify":
        verify_usd_files(
            args.sequence_graph,
            args.usd_root,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
