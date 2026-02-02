"""Tests for RocksDB and SequenceGraph integration.

Run with: python tests/test_rocksdb_integration.py
"""

import sys
from pathlib import Path
import tempfile
import shutil
import numpy as np
import networkx as nx
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from navisim_lab.data import RocksDBManager, get_db, reset_db
from navisim_lab.scene import SequenceGraph, SceneGraph


def test_rocksdb_manager():
    """Test RocksDB manager basic functionality."""
    print("\n" + "="*60)
    print("Test 1: RocksDB Manager")
    print("="*60)

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_rocksdb"
        db = RocksDBManager(db_path)

        # Test save and retrieve
        db.save("test_key", "test_value")
        value = db.get("test_key")
        assert value == "test_value", f"Expected 'test_value', got {value}"
        print("✓ Basic save/get works")

        # Test dict storage
        test_dict = {"key1": "value1", "key2": [1, 2, 3]}
        db.save("dict_key", test_dict)
        retrieved = db.get("dict_key")
        assert retrieved == test_dict, f"Expected {test_dict}, got {retrieved}"
        print("✓ Dict serialization works")

        # Test exists
        assert db.exists("test_key")
        assert not db.exists("nonexistent_key")
        print("✓ Exists check works")

        # Test delete
        db.delete("test_key")
        assert not db.exists("test_key")
        print("✓ Delete works")

        # Clean up
        db.close()

    print("✓ All RocksDB manager tests passed!")


def test_section_metadata():
    """Test section metadata storage."""
    print("\n" + "="*60)
    print("Test 2: Section Metadata")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_rocksdb"
        db = RocksDBManager(db_path)

        # Save section metadata
        db.save_section_metadata(
            section_id="seq001/sector001",
            usd_path="/path/to/section.usd",
            bounds=[[0, 0, 0], [50, 50, 10]],
            center=[25, 25, 5],
            metadata={"type": "warehouse"}
        )

        # Retrieve metadata
        metadata = db.get_section_metadata("seq001/sector001")
        assert metadata is not None
        assert metadata["usd_path"] == "/path/to/section.usd"
        assert metadata["bounds"] == [[0, 0, 0], [50, 50, 10]]
        assert metadata["center"] == [25, 25, 5]
        assert metadata["metadata"]["type"] == "warehouse"
        print("✓ Section metadata storage works")

        # List sections
        sections = db.list_sections()
        assert "seq001/sector001" in sections
        print(f"✓ List sections works: {sections}")

        db.close()

    print("✓ All section metadata tests passed!")


def test_sequence_graph():
    """Test SequenceGraph loading and conversion."""
    print("\n" + "="*60)
    print("Test 3: SequenceGraph")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock sequence graph
        graph = nx.Graph()
        graph.add_node("seq001", sectors=["sector001", "sector002"])
        graph.add_node("seq002", sectors=["sector003"])
        graph.add_edge("seq001", "seq002")

        # Save as pickle
        graph_path = Path(tmpdir) / "sequence_graph.gpickle"
        with open(graph_path, "wb") as f:
            pickle.dump(graph, f)

        # Create RocksDB and populate
        db_path = Path(tmpdir) / "test_rocksdb"
        db = RocksDBManager(db_path)

        # Add metadata for sections
        db.save_section_metadata(
            section_id="seq001/sector001",
            usd_path="assets/seq001_sector001.usd",
            bounds=[[0, 0, 0], [50, 50, 10]],
            center=[25, 25, 5],
        )

        # Load sequence graph (without using global singleton)
        seq_graph = SequenceGraph(graph_path, use_rocksdb=False)

        # Test basic queries
        sequence_ids = seq_graph.get_sequence_ids()
        assert "seq001" in sequence_ids
        assert "seq002" in sequence_ids
        print(f"✓ Got sequence IDs: {sequence_ids}")

        sectors = seq_graph.get_sequence("seq001")
        assert sectors == ["sector001", "sector002"]
        print(f"✓ Got sectors for seq001: {sectors}")

        # Test conversion to SceneGraph
        usd_root = Path(tmpdir) / "usd"
        usd_root.mkdir()

        scene_graph = seq_graph.to_scene_graph(
            usd_root=usd_root,
            default_bounds=[[0, 0, 0], [50, 50, 10]]
        )

        assert len(scene_graph) > 0
        print(f"✓ Converted to SceneGraph: {len(scene_graph)} sections")

        db.close()

    print("✓ All SequenceGraph tests passed!")


def test_scene_graph_spatial_queries():
    """Test SceneGraph spatial queries."""
    print("\n" + "="*60)
    print("Test 4: SceneGraph Spatial Queries")
    print("="*60)

    scene_graph = SceneGraph()

    # Add test sections
    scene_graph.add_section(
        section_id="sec001",
        usd_path=Path("test.usd"),
        bounds=np.array([[0, 0, 0], [50, 50, 10]]),
        center=np.array([25, 25, 5]),
        neighbors=["sec002"]
    )

    scene_graph.add_section(
        section_id="sec002",
        usd_path=Path("test2.usd"),
        bounds=np.array([[50, 0, 0], [100, 50, 10]]),
        center=np.array([75, 25, 5]),
        neighbors=["sec001"]
    )

    # Test point containment
    section_id = scene_graph.find_section_containing_point([25, 25, 5])
    assert section_id == "sec001"
    print(f"✓ Point containment works: found {section_id}")

    # Test neighbors
    neighbors = scene_graph.get_neighbors("sec001", depth=1)
    assert "sec002" in neighbors
    print(f"✓ Neighbor query works: {neighbors}")

    # Test nearest sections
    nearest = scene_graph.get_k_nearest_sections([25, 25, 5], k=2)
    assert len(nearest) == 2
    assert nearest[0][0] == "sec001"  # Closest should be sec001
    print(f"✓ K-nearest works: {[sec_id for sec_id, _ in nearest]}")

    # Test within radius
    within = scene_graph.get_sections_within_radius([25, 25, 5], radius=60)
    assert "sec001" in within
    print(f"✓ Within radius works: {within}")

    # Test loading strategy
    to_load, to_unload = scene_graph.compute_loading_strategy(
        current_position=np.array([25, 25, 5]),
        max_loaded_sections=5,
        load_radius=60.0,
        neighbor_depth=1
    )
    assert "sec001" in to_load
    print(f"✓ Loading strategy works: load={list(to_load)}, unload={list(to_unload)}")

    print("✓ All spatial query tests passed!")


def main():
    """Run all tests."""
    print("="*60)
    print("RocksDB and SequenceGraph Integration Tests")
    print("="*60)

    try:
        test_rocksdb_manager()
        test_section_metadata()
        test_sequence_graph()
        test_scene_graph_spatial_queries()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
