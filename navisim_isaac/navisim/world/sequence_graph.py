"""
SequenceGraph Module

Handles the NaviSim SequenceGraph containing usdz and height field data.
"""

import pickle
from collections import defaultdict
from typing import Dict, List, Any, Optional
import networkx as nx

from .sector import Sector


class SequenceGraph:
    """
    Manages the sequence graph from NaviSim containing scene representations.

    This class wraps a NetworkX graph loaded from a pickle file and provides
    convenient access to sequences and sectors.

    The graph structure:
    - Nodes represent sequences (sequence IDs)
    - Node data contains 'sectors': list of sector IDs for that sequence
    - Each sector contains height field, USDZ, and boundary polygon data
    """

    def __init__(self, path: str):
        """
        Initialize SequenceGraph from a pickle file.

        Args:
            path: Path to the pickled NetworkX graph file
        """
        self.graph = self._load_sequence_graph(path)
        self.sequences = self._load_get_sequences(self.graph)
        self.current_node = None

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying NetworkX graph.

        This allows direct access to NetworkX methods like nodes(), edges(), etc.

        Args:
            name: Attribute name

        Returns:
            Attribute from the underlying graph
        """
        return getattr(self.graph, name)

    def get_sequence(self, seq_id: str) -> List[Sector]:
        """
        Get all sectors for a given sequence.

        Args:
            seq_id: Sequence identifier

        Returns:
            List of Sector objects for the sequence
        """
        return self.sequences[seq_id]

    def get_sequence_ids(self) -> List[str]:
        """
        Get all sequence IDs in the graph.

        Returns:
            List of sequence identifiers
        """
        return list(self.sequences.keys())

    def set_current_node(self, seq_id: str) -> None:
        """
        Set the current navigation node.

        Args:
            seq_id: Sequence identifier to set as current
        """
        if seq_id not in self.sequences:
            raise ValueError(f"Sequence {seq_id} not found in graph")
        self.current_node = seq_id

    def get_current_node(self) -> Optional[str]:
        """
        Get the current navigation node.

        Returns:
            Current sequence ID or None
        """
        return self.current_node

    def get_neighbors(self, seq_id: str) -> List[str]:
        """
        Get neighboring sequences for navigation.

        Args:
            seq_id: Sequence identifier

        Returns:
            List of neighboring sequence IDs
        """
        return list(self.graph.neighbors(seq_id))

    def request_next_node(self, current_seq_id: Optional[str] = None) -> Optional[str]:
        """
        Request the next navigation node.

        Args:
            current_seq_id: Current sequence ID (uses self.current_node if None)

        Returns:
            Next sequence ID or None if no neighbors
        """
        seq_id = current_seq_id or self.current_node
        if seq_id is None:
            raise ValueError("No current node set. Call set_current_node first.")

        neighbors = self.get_neighbors(seq_id)
        if neighbors:
            next_node = neighbors[0]  # Simple strategy: take first neighbor
            self.current_node = next_node
            return next_node
        return None

    def _load_get_sequences(self, sequence_graph: nx.Graph) -> Dict[str, List[Sector]]:
        """
        Load all sequences from the graph.

        Args:
            sequence_graph: NetworkX graph containing sequence data

        Returns:
            Dictionary mapping sequence IDs to lists of Sectors
        """
        sequences = defaultdict(list)
        for node, data in sequence_graph.nodes(data=True):
            sequences[node] = self._load_sectors(node, data['sectors'])
        return sequences

    def _load_sectors(self, seq_id: str, sector_ids: List[str]) -> List[Sector]:
        """
        Create Sector objects and link them in sequence.

        Args:
            seq_id: Sequence identifier
            sector_ids: List of sector IDs in the sequence

        Returns:
            List of linked Sector objects
        """
        sectors = [Sector(seq_id, sec_id) for sec_id in sector_ids]
        for i, sector in enumerate(sectors):
            if i > 0:
                sector.prev = sectors[i - 1]
            if i < len(sectors) - 1:
                sector.next = sectors[i + 1]
        return sectors

    def _load_sequence_graph(self, graph_path: str) -> nx.Graph:
        """
        Load NetworkX graph from pickle file.

        Args:
            graph_path: Path to the pickle file

        Returns:
            Loaded NetworkX graph

        Raises:
            FileNotFoundError: If the graph file is not found
        """
        try:
            with open(graph_path, 'rb') as f:
                sequence_graph = pickle.load(f)
            print(f"Loaded sequence graph from {graph_path}")
            return sequence_graph
        except FileNotFoundError:
            raise FileNotFoundError(f'{graph_path} is not available')
