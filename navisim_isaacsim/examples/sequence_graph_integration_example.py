"""Example: Using SequenceGraph and RocksDB with Isaac Sim.

This script demonstrates how to:
1. Load a NaviSim sequence graph
2. Use RocksDB for USD metadata storage
3. Create a SceneGraph for dynamic loading
4. Integrate with DynamicSceneManager in Isaac Sim
"""

import logging
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Isaac Sim AppLauncher must be first!
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# Now import Isaac and NaviSim modules
from omni.isaac.core import World
from omni.isaac.core.utils.stage import get_current_stage

from navisim_lab.scene import SequenceGraph, SceneGraph, DynamicSceneManager
from navisim_lab.data import get_db, reset_db


def example_setup_rocksdb():
    """Example: Set up RocksDB with USD metadata."""
    logger.info("=" * 60)
    logger.info("Example 1: Setting up RocksDB")
    logger.info("=" * 60)

    # Path to your NaviSim sequence graph
    sequence_graph_path = Path("assets/sequence_graph.gpickle")

    # Directory containing your USD files
    usd_root = Path("assets/usd_sections")

    # Check if files exist
    if not sequence_graph_path.exists():
        logger.error(f"Sequence graph not found: {sequence_graph_path}")
        logger.info("Please download sequence_graph.gpickle to assets/ directory")
        return False

    if not usd_root.exists():
        logger.warning(f"USD root not found: {usd_root}")
        logger.info("Creating directory. You'll need to add USD files.")
        usd_root.mkdir(parents=True, exist_ok=True)

    # Load sequence graph
    seq_graph = SequenceGraph(sequence_graph_path, use_rocksdb=True)

    logger.info(f"Loaded sequence graph: {seq_graph}")
    logger.info(f"Sequences: {seq_graph.get_sequence_ids()[:5]}...")  # Show first 5

    # Populate RocksDB with USD metadata
    # This scans usd_root for USD files and stores their metadata
    seq_graph.populate_rocksdb_from_directory(
        usd_root=usd_root,
        default_bounds=[[0, 0, 0], [50, 50, 10]],  # Default bounds
    )

    # Verify what's in RocksDB
    db = get_db()
    sections = db.list_sections()
    logger.info(f"RocksDB now contains {len(sections)} sections")

    if sections:
        # Show first section as example
        first_section = sections[0]
        metadata = db.get_section_metadata(first_section)
        logger.info(f"Example section: {first_section}")
        logger.info(f"  USD path: {metadata.get('usd_path')}")
        logger.info(f"  Bounds: {metadata.get('bounds')}")
        logger.info(f"  Center: {metadata.get('center')}")

    return True


def example_convert_to_scene_graph():
    """Example: Convert SequenceGraph to SceneGraph."""
    logger.info("=" * 60)
    logger.info("Example 2: Converting to SceneGraph")
    logger.info("=" * 60)

    sequence_graph_path = Path("assets/sequence_graph.gpickle")

    if not sequence_graph_path.exists():
        logger.error(f"Sequence graph not found: {sequence_graph_path}")
        return None

    # Load sequence graph (will use RocksDB for USD metadata)
    seq_graph = SequenceGraph(sequence_graph_path, use_rocksdb=True)

    # Convert to SceneGraph format
    scene_graph = seq_graph.to_scene_graph(
        usd_root=Path("assets/usd_sections"),  # Fallback if no RocksDB
        default_bounds=[[0, 0, 0], [50, 50, 10]],
    )

    logger.info(f"Created scene graph: {scene_graph}")
    logger.info(f"Total sections: {len(scene_graph)}")
    logger.info(f"Graph edges: {scene_graph.graph.number_of_edges()}")

    # Save for later use
    output_path = Path("assets/scene_graph.pkl")
    scene_graph.to_pickle(output_path)
    logger.info(f"Saved to: {output_path}")

    return scene_graph


def example_dynamic_loading_simulation(scene_graph: SceneGraph):
    """Example: Dynamic loading with Isaac Sim."""
    logger.info("=" * 60)
    logger.info("Example 3: Dynamic Loading in Isaac Sim")
    logger.info("=" * 60)

    # Create Isaac Sim world
    world = World(stage_units_in_meters=1.0)
    stage = get_current_stage()

    # Create dynamic scene manager
    manager = DynamicSceneManager(
        scene_graph=scene_graph,
        stage=stage,
        max_loaded_sections=9,  # Keep up to 9 sections loaded
        load_radius=50.0,  # Load sections within 50m
        neighbor_depth=1,  # Include 1-hop neighbors
    )

    logger.info(f"Created DynamicSceneManager: {manager}")

    # Simulate robot movement
    logger.info("\nSimulating robot navigation...")

    # Example robot trajectory
    robot_positions = [
        np.array([10.0, 10.0, 0.0]),
        np.array([25.0, 25.0, 0.0]),
        np.array([40.0, 40.0, 0.0]),
        np.array([30.0, 50.0, 0.0]),
    ]

    for i, position in enumerate(robot_positions):
        logger.info(f"\nStep {i+1}: Robot at {position}")

        # Update scene based on robot position
        result = manager.update(position)

        logger.info(f"  Loaded sections: {result['loaded']}")
        logger.info(f"  Unloaded sections: {result['unloaded']}")

        current_section = manager.get_current_section(position)
        logger.info(f"  Current section: {current_section}")

        stats = manager.get_stats()
        logger.info(f"  Total loaded: {stats['loaded_sections']}/{stats['total_sections']}")

    logger.info(f"\nFinal stats: {manager.get_stats()}")

    # Cleanup
    manager.reset()


def example_query_sequence_info():
    """Example: Query sequence and sector information."""
    logger.info("=" * 60)
    logger.info("Example 4: Querying Sequence Information")
    logger.info("=" * 60)

    sequence_graph_path = Path("assets/sequence_graph.gpickle")

    if not sequence_graph_path.exists():
        logger.error(f"Sequence graph not found: {sequence_graph_path}")
        return

    seq_graph = SequenceGraph(sequence_graph_path, use_rocksdb=True)

    # Get all sequences
    sequence_ids = seq_graph.get_sequence_ids()
    logger.info(f"Total sequences: {len(sequence_ids)}")

    # Show first sequence
    if sequence_ids:
        seq_id = sequence_ids[0]
        sectors = seq_graph.get_sequence(seq_id)

        logger.info(f"\nSequence: {seq_id}")
        logger.info(f"  Sectors: {sectors}")

        # Get metadata for first sector
        if sectors:
            sector_id = sectors[0]
            metadata = seq_graph.get_sector_metadata(seq_id, sector_id)

            if metadata:
                logger.info(f"\nSector {sector_id} metadata:")
                logger.info(f"  USD path: {metadata.get('usd_path')}")
                logger.info(f"  Bounds: {metadata.get('bounds')}")
                logger.info(f"  Center: {metadata.get('center')}")
            else:
                logger.info(f"No metadata found for {seq_id}/{sector_id}")


def main():
    """Run all examples."""
    logger.info("NaviSim SequenceGraph + RocksDB Integration Examples")
    logger.info("=" * 60)

    try:
        # Example 1: Set up RocksDB
        success = example_setup_rocksdb()

        if not success:
            logger.error("Setup failed. Check that you have sequence_graph.gpickle")
            return

        # Example 2: Convert to SceneGraph
        scene_graph = example_convert_to_scene_graph()

        if scene_graph is None:
            logger.error("Scene graph conversion failed")
            return

        # Example 3: Dynamic loading simulation
        example_dynamic_loading_simulation(scene_graph)

        # Example 4: Query information
        example_query_sequence_info()

        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during examples: {e}", exc_info=True)

    finally:
        # Cleanup
        simulation_app.close()


if __name__ == "__main__":
    main()
