"""Sequence graph integration for NaviSim Isaac Lab.

This module provides compatibility with NaviSim's sequence_graph.gpickle format
while integrating with RocksDB for USD asset storage and SceneGraph for streaming.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional
import networkx as nx
import numpy as np

from .scene_graph import SceneGraph, SceneSection
from ..data import get_db

logger = logging.getLogger(__name__)


class SequenceGraph:
    """NaviSim sequence graph adapter for Isaac Lab.

    Loads NaviSim's sequence_graph.gpickle and provides access to sequences
    and sectors, with integration to RocksDB for USD asset metadata.

    Compatible with original NaviSim SequenceGraph API.
    """

    def __init__(self, graph_path: Path, use_rocksdb: bool = True):
        """Initialize sequence graph from pickle file.

        Args:
            graph_path: Path to sequence_graph.gpickle
            use_rocksdb: Whether to load USD metadata from RocksDB
        """
        self.graph_path = Path(graph_path)
        self.use_rocksdb = use_rocksdb
        self.graph = self._load_sequence_graph(graph_path)
        self.sequences = self._load_sequences(self.graph)
        self.db = get_db() if use_rocksdb else None

        logger.info(
            f"Loaded sequence graph from {graph_path}: "
            f"{len(self.sequences)} sequences, "
            f"{self.graph.number_of_nodes()} nodes"
        )

    def __getattr__(self, name):
        """Delegate attribute access to underlying NetworkX graph."""
        return getattr(self.graph, name)

    def get_sequence(self, seq_id: str) -> List[str]:
        """Get list of sector IDs for a sequence.

        Args:
            seq_id: Sequence identifier

        Returns:
            List of sector IDs in order
        """
        return self.sequences.get(seq_id, [])

    def get_sequence_ids(self) -> List[str]:
        """Get all sequence IDs.

        Returns:
            List of sequence identifiers
        """
        return list(self.sequences.keys())

    def get_sector_usd_path(self, seq_id: str, sector_id: str) -> Optional[Path]:
        """Get USD file path for a sector from RocksDB.

        Args:
            seq_id: Sequence identifier
            sector_id: Sector identifier

        Returns:
            Path to USD file or None if not found
        """
        if not self.db:
            logger.warning("RocksDB not enabled, cannot retrieve USD path")
            return None

        # Construct section ID (format: seq_id/sector_id)
        section_id = f"{seq_id}/{sector_id}"

        try:
            metadata = self.db.get_section_metadata(section_id)
            if metadata:
                usd_path = metadata.get("usd_path")
                if usd_path:
                    return Path(usd_path)
        except KeyError:
            logger.debug(f"No USD metadata found for {section_id}")

        return None

    def get_sector_metadata(self, seq_id: str, sector_id: str) -> Optional[Dict]:
        """Get metadata for a sector from RocksDB.

        Args:
            seq_id: Sequence identifier
            sector_id: Sector identifier

        Returns:
            Metadata dict with bounds, center, etc.
        """
        if not self.db:
            return None

        section_id = f"{seq_id}/{sector_id}"

        try:
            return self.db.get_section_metadata(section_id)
        except KeyError:
            return None

    def to_scene_graph(
        self,
        usd_root: Optional[Path] = None,
        default_bounds: Optional[List] = None,
    ) -> SceneGraph:
        """Convert this sequence graph to a SceneGraph for dynamic loading.

        Args:
            usd_root: Root directory for USD files (if not using RocksDB paths)
            default_bounds: Default bounds if not in metadata [[min], [max]]

        Returns:
            SceneGraph instance ready for DynamicSceneManager
        """
        scene_graph = SceneGraph(self.graph)

        default_bounds = default_bounds or [[0, 0, 0], [50, 50, 10]]

        logger.info("Converting SequenceGraph to SceneGraph...")

        # Iterate through all sequences and sectors
        for seq_id, sector_ids in self.sequences.items():
            for sector_id in sector_ids:
                section_id = f"{seq_id}/{sector_id}"

                # Try to get metadata from RocksDB
                metadata = self.get_sector_metadata(seq_id, sector_id)

                if metadata:
                    usd_path = Path(metadata["usd_path"])
                    bounds = np.array(metadata.get("bounds", default_bounds))
                    center = np.array(metadata.get("center", [25, 25, 5]))
                    extra_metadata = metadata.get("metadata", {})
                else:
                    # Fallback: construct path and use defaults
                    if usd_root:
                        usd_path = usd_root / f"{seq_id}_{sector_id}.usd"
                    else:
                        usd_path = Path(f"assets/sections/{seq_id}_{sector_id}.usd")

                    bounds = np.array(default_bounds)
                    center = np.array(
                        [
                            (bounds[0][0] + bounds[1][0]) / 2,
                            (bounds[0][1] + bounds[1][1]) / 2,
                            (bounds[0][2] + bounds[1][2]) / 2,
                        ]
                    )
                    extra_metadata = {}

                    logger.debug(
                        f"No metadata for {section_id}, using defaults"
                    )

                # Get neighbors from graph
                neighbors = []
                if seq_id in self.graph:
                    neighbors = [
                        f"{neighbor_seq}/{sector}"
                        for neighbor_seq in self.graph.neighbors(seq_id)
                        for sector in self.get_sequence(neighbor_seq)
                    ]

                # Add section to scene graph
                scene_graph.add_section(
                    section_id=section_id,
                    usd_path=usd_path,
                    bounds=bounds,
                    center=center,
                    metadata=extra_metadata,
                    neighbors=neighbors,
                )

        logger.info(
            f"Converted to SceneGraph: {len(scene_graph)} sections, "
            f"{scene_graph.graph.number_of_edges()} edges"
        )

        return scene_graph

    def populate_rocksdb_from_directory(
        self,
        usd_root: Path,
        default_bounds: Optional[List] = None,
        auto_compute_bounds: bool = False,
    ):
        """Populate RocksDB with USD metadata by scanning a directory.

        This is a utility to set up RocksDB if you have USD files but no metadata.

        Args:
            usd_root: Root directory containing USD files
            default_bounds: Default bounds to use [[min], [max]]
            auto_compute_bounds: Try to compute bounds from USD (not implemented)
        """
        if not self.db:
            logger.error("RocksDB not enabled")
            return

        usd_root = Path(usd_root)
        default_bounds = default_bounds or [[0, 0, 0], [50, 50, 10]]

        logger.info(f"Populating RocksDB from {usd_root}...")

        count = 0

        for seq_id, sector_ids in self.sequences.items():
            for sector_id in sector_ids:
                section_id = f"{seq_id}/{sector_id}"

                # Look for USD file
                usd_path = usd_root / f"{seq_id}_{sector_id}.usd"

                if not usd_path.exists():
                    # Try alternative naming
                    usd_path = usd_root / seq_id / f"{sector_id}.usd"

                if not usd_path.exists():
                    logger.warning(f"USD file not found for {section_id}")
                    continue

                # Compute or use default bounds
                bounds = default_bounds
                center = [
                    (bounds[0][0] + bounds[1][0]) / 2,
                    (bounds[0][1] + bounds[1][1]) / 2,
                    (bounds[0][2] + bounds[1][2]) / 2,
                ]

                # Save to RocksDB
                self.db.save_section_metadata(
                    section_id=section_id,
                    usd_path=str(usd_path),
                    bounds=bounds,
                    center=center,
                    metadata={
                        "sequence_id": seq_id,
                        "sector_id": sector_id,
                    },
                )

                count += 1

        logger.info(f"Populated RocksDB with {count} section metadata entries")

    def _load_sequences(self, graph: nx.Graph) -> Dict[str, List[str]]:
        """Extract sequences from NetworkX graph.

        Args:
            graph: NetworkX graph with node data

        Returns:
            Dict mapping sequence_id to list of sector_ids
        """
        sequences = {}

        for node_id, node_data in graph.nodes(data=True):
            # Extract sectors from node data
            # Adjust this based on your actual data structure
            sectors = node_data.get("sectors", [])

            if not sectors and isinstance(node_data, dict):
                # Try alternative keys
                sectors = node_data.get("sector_ids", [])

            sequences[node_id] = sectors

        return sequences

    def _load_sequence_graph(self, graph_path: Path) -> nx.Graph:
        """Load sequence graph from pickle file.

        Args:
            graph_path: Path to sequence_graph.gpickle

        Returns:
            NetworkX graph

        Raises:
            FileNotFoundError: If graph file doesn't exist
        """
        try:
            with open(graph_path, "rb") as f:
                graph = pickle.load(f)

            if not isinstance(graph, nx.Graph):
                logger.warning(
                    f"Loaded object is not a NetworkX graph: {type(graph)}"
                )

            return graph

        except FileNotFoundError:
            logger.error(f"Sequence graph file not found: {graph_path}")
            raise

    def __repr__(self) -> str:
        return (
            f"SequenceGraph("
            f"sequences={len(self.sequences)}, "
            f"nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()})"
        )
