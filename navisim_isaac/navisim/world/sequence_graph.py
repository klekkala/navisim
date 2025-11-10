"""
SequenceGraph Module

Handles the NaviSim SequenceGraph containing USD scene data.
"""

import pickle
from typing import Dict, Optional
import networkx as nx

from .sector import Sector


class SequenceGraph:
    """
    Manages the navigation graph from NaviSim containing scene representations.

    This class wraps a NetworkX graph loaded from a pickle file and provides
    convenient access to sectors (nodes) in the navigation environment.

    The graph structure:
    - Each node represents a single sector (navigation environment segment)
    - Each node has 'usd_path' pointing to a .usd file containing:
      - Gaussian splat scene (for RL visual input)
      - Height field mesh (for physics/collision detection)
    - Edges represent navigable connections between sectors
    """

    def __init__(self, path: str):
        """
        Initialize SequenceGraph from a pickle file.

        Args:
            path: Path to the pickled NetworkX graph file
        """
        self.graph = self._load_sequence_graph(path)
        self.sectors = self._load_sectors(self.graph)
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

    def get_sector(self, sector_id: str) -> Sector:
        """
        Get a sector by its ID.

        Args:
            sector_id: Sector identifier (node ID in graph)

        Returns:
            Sector object

        Raises:
            KeyError: If sector not found
        """
        if sector_id not in self.sectors:
            raise KeyError(f"Sector {sector_id} not found in graph")
        return self.sectors[sector_id]

    def get_sector_ids(self) -> list:
        """
        Get all sector IDs in the graph.

        Returns:
            List of sector identifiers
        """
        return list(self.sectors.keys())

    def set_current_node(self, sector_id: str) -> None:
        """
        Set the current navigation node (sector).

        Args:
            sector_id: Sector identifier to set as current

        Raises:
            ValueError: If sector not found
        """
        if sector_id not in self.sectors:
            raise ValueError(f"Sector {sector_id} not found in graph")
        self.current_node = sector_id

    def get_current_node(self) -> Optional[str]:
        """
        Get the current navigation node (sector ID).

        Returns:
            Current sector ID or None
        """
        return self.current_node

    def get_current_sector(self) -> Optional[Sector]:
        """
        Get the current sector object.

        Returns:
            Current Sector or None
        """
        if self.current_node is None:
            return None
        return self.sectors.get(self.current_node)

    def get_neighbors(self, sector_id: str) -> list:
        """
        Get neighboring sectors for navigation.

        Args:
            sector_id: Sector identifier

        Returns:
            List of neighboring sector IDs
        """
        return list(self.graph.neighbors(sector_id))

    def request_next_node(self, current_sector_id: Optional[str] = None) -> Optional[str]:
        """
        Request the next navigation node (sector).

        Args:
            current_sector_id: Current sector ID (uses self.current_node if None)

        Returns:
            Next sector ID or None if no neighbors

        Raises:
            ValueError: If no current node set
        """
        sector_id = current_sector_id or self.current_node
        if sector_id is None:
            raise ValueError("No current node set. Call set_current_node first.")

        neighbors = self.get_neighbors(sector_id)
        if neighbors:
            next_node = neighbors[0]  # Simple strategy: take first neighbor
            self.current_node = next_node
            return next_node
        return None

    def _load_sectors(self, graph: nx.Graph) -> Dict[str, Sector]:
        """
        Load all sectors from the graph nodes.

        Args:
            graph: NetworkX graph containing sector data

        Returns:
            Dictionary mapping sector IDs to Sector objects
        """
        sectors = {}
        for node_id, data in graph.nodes(data=True):
            # Extract USD path from node data
            usd_path = data.get('usd_path', None)
            if usd_path is None:
                print(f"Warning: Node {node_id} missing 'usd_path', skipping")
                continue

            # Create Sector with USD path
            sector = Sector(
                sector_id=node_id,
                usd_path=usd_path,
                metadata=data
            )
            sectors[node_id] = sector

        print(f"Loaded {len(sectors)} sectors from graph")
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
                graph = pickle.load(f)
            print(f"Loaded navigation graph from {graph_path}")
            print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            return graph
        except FileNotFoundError:
            raise FileNotFoundError(f'Graph file not found: {graph_path}')

    def unload_all_sectors(self) -> None:
        """
        Unload all sector data to free memory.
        """
        for sector in self.sectors.values():
            sector.unload_all()
