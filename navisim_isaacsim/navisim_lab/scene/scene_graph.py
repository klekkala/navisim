"""Scene graph manager for dynamic USD section loading.

This module manages the spatial organization of USD sections using a NetworkX graph,
enabling efficient streaming of environment sections as robots traverse the map.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class SceneSection:
    """Represents a single section of the map with its USD file and metadata."""

    def __init__(
        self,
        section_id: str,
        usd_path: Path,
        bounds: np.ndarray,
        center: np.ndarray,
        metadata: Optional[Dict] = None,
    ):
        """Initialize a scene section.

        Args:
            section_id: Unique identifier for this section
            usd_path: Path to the USD file for this section
            bounds: Bounding box as [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            center: Center point of the section [x, y, z]
            metadata: Optional metadata (e.g., type, features, difficulty)
        """
        self.section_id = section_id
        self.usd_path = Path(usd_path)
        self.bounds = np.array(bounds)
        self.center = np.array(center)
        self.metadata = metadata or {}
        self.is_loaded = False

    @property
    def min_bounds(self) -> np.ndarray:
        """Get minimum bounds [min_x, min_y, min_z]."""
        return self.bounds[0]

    @property
    def max_bounds(self) -> np.ndarray:
        """Get maximum bounds [max_x, max_y, max_z]."""
        return self.bounds[1]

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is within this section's bounds (3D).

        Args:
            point: Point to check [x, y, z]

        Returns:
            True if point is inside bounds
        """
        point = np.array(point)
        return np.all(point >= self.min_bounds) and np.all(point <= self.max_bounds)

    def contains_point_xy(self, point: np.ndarray) -> bool:
        """Check if a point is within this section's XY footprint (ignores Z).

        Use this for robots navigating on a flat ground plane, where the robot's
        Z coordinate may not fall within the sector's Z bounds.

        Args:
            point: Point to check [x, y, ...] (Z ignored)

        Returns:
            True if (x, y) is inside the section's XY bounds
        """
        p = np.array(point)
        return (
            p[0] >= self.min_bounds[0] and p[0] <= self.max_bounds[0]
            and p[1] >= self.min_bounds[1] and p[1] <= self.max_bounds[1]
        )

    def distance_to_point(self, point: np.ndarray) -> float:
        """Calculate distance from section center to a point.

        Args:
            point: Point coordinates [x, y, z]

        Returns:
            Euclidean distance
        """
        return np.linalg.norm(self.center - np.array(point))

    def __repr__(self) -> str:
        return f"SceneSection(id={self.section_id}, loaded={self.is_loaded})"


class SceneGraph:
    """Manages the spatial graph of scene sections for dynamic loading.

    The graph nodes represent sections, and edges represent spatial adjacency.
    This enables efficient queries for nearby sections and path planning.
    """

    def __init__(self, graph_data: Optional[nx.Graph] = None):
        """Initialize the scene graph.

        Args:
            graph_data: Optional pre-built NetworkX graph with section data
        """
        self.graph = graph_data if graph_data is not None else nx.Graph()
        self.sections: Dict[str, SceneSection] = {}
        self.loaded_sections: Set[str] = set()

        # Spatial index for fast proximity queries
        self._section_centers = []
        self._section_ids = []

    def add_section(
        self,
        section_id: str,
        usd_path: Path,
        bounds: np.ndarray,
        center: np.ndarray,
        metadata: Optional[Dict] = None,
        neighbors: Optional[List[str]] = None,
    ) -> SceneSection:
        """Add a section to the graph.

        Args:
            section_id: Unique section identifier
            usd_path: Path to USD file
            bounds: Bounding box [[min], [max]]
            center: Center point [x, y, z]
            metadata: Optional metadata
            neighbors: Optional list of neighboring section IDs

        Returns:
            Created SceneSection object
        """
        section = SceneSection(section_id, usd_path, bounds, center, metadata)
        self.sections[section_id] = section
        self.graph.add_node(section_id, section=section)

        # Update spatial index
        self._section_centers.append(center)
        self._section_ids.append(section_id)

        # Add edges to neighbors
        if neighbors:
            for neighbor_id in neighbors:
                if neighbor_id in self.sections:
                    self.graph.add_edge(section_id, neighbor_id)

        logger.debug(f"Added section: {section_id} with {len(neighbors or [])} neighbors")
        return section

    def get_section(self, section_id: str) -> Optional[SceneSection]:
        """Get a section by ID.

        Args:
            section_id: Section identifier

        Returns:
            SceneSection or None if not found
        """
        return self.sections.get(section_id)

    def find_section_containing_point(self, point: np.ndarray) -> Optional[str]:
        """Find which section contains a given point using XY footprint only.

        XY-only matching is used because robots navigate on a flat ground plane
        and their Z coordinate may not fall within a sector's 3D Z bounds.

        Args:
            point: Point coordinates [x, y, z] (Z ignored for containment)

        Returns:
            Section ID or None if point is not in any section
        """
        for section_id, section in self.sections.items():
            if section.contains_point_xy(point):
                return section_id
        return None

    def get_neighbors(self, section_id: str, depth: int = 1) -> Set[str]:
        """Get neighboring sections up to a certain depth.

        Args:
            section_id: Center section ID
            depth: How many hops to include (1 = immediate neighbors)

        Returns:
            Set of neighboring section IDs
        """
        if section_id not in self.graph:
            return set()

        neighbors = set()
        for d in range(1, depth + 1):
            # Get all nodes at distance d
            nodes_at_depth = nx.single_source_shortest_path_length(
                self.graph, section_id, cutoff=d
            )
            neighbors.update(node for node, dist in nodes_at_depth.items() if dist == d)

        return neighbors

    def get_k_nearest_sections(
        self, point: np.ndarray, k: int = 5, exclude_loaded: bool = False
    ) -> List[Tuple[str, float]]:
        """Get K nearest sections to a point.

        Args:
            point: Query point [x, y, z]
            k: Number of nearest sections to return
            exclude_loaded: Whether to exclude already loaded sections

        Returns:
            List of (section_id, distance) tuples, sorted by distance
        """
        point = np.array(point)

        distances = []
        for section_id, section in self.sections.items():
            if exclude_loaded and section_id in self.loaded_sections:
                continue
            dist = section.distance_to_point(point)
            distances.append((section_id, dist))

        # Sort by distance and return top k
        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def get_sections_within_radius(
        self, point: np.ndarray, radius: float
    ) -> List[str]:
        """Get all sections within a radius of a point.

        Args:
            point: Center point [x, y, z]
            radius: Search radius

        Returns:
            List of section IDs within radius
        """
        point = np.array(point)
        result = []

        for section_id, section in self.sections.items():
            if section.distance_to_point(point) <= radius:
                result.append(section_id)

        return result

    def compute_loading_strategy(
        self,
        current_position: np.ndarray,
        max_loaded_sections: int = 9,
        load_radius: float = 50.0,
        neighbor_depth: int = 1,
    ) -> Tuple[Set[str], Set[str]]:
        """Compute which sections to load/unload based on robot position.

        Args:
            current_position: Current robot position [x, y, z]
            max_loaded_sections: Maximum number of sections to keep loaded
            load_radius: Radius around robot to keep loaded
            neighbor_depth: How many neighbor hops to include

        Returns:
            Tuple of (sections_to_load, sections_to_unload)
        """
        # Find current section
        current_section_id = self.find_section_containing_point(current_position)

        if current_section_id is None:
            logger.warning(
                f"Robot at {current_position} is not in any known section"
            )
            return set(), set()

        # Determine sections to keep loaded
        sections_to_keep = {current_section_id}

        # Add immediate neighbors from graph
        sections_to_keep.update(self.get_neighbors(current_section_id, neighbor_depth))

        # Add sections within radius
        nearby = self.get_sections_within_radius(current_position, load_radius)
        sections_to_keep.update(nearby)

        # Limit to max_loaded_sections by prioritizing by distance
        if len(sections_to_keep) > max_loaded_sections:
            distances = [
                (sid, self.sections[sid].distance_to_point(current_position))
                for sid in sections_to_keep
            ]
            distances.sort(key=lambda x: x[1])
            sections_to_keep = {sid for sid, _ in distances[:max_loaded_sections]}

        # Determine what to load and unload
        sections_to_load = sections_to_keep - self.loaded_sections
        sections_to_unload = self.loaded_sections - sections_to_keep

        return sections_to_load, sections_to_unload

    def mark_loaded(self, section_id: str):
        """Mark a section as loaded.

        Args:
            section_id: Section to mark
        """
        if section_id in self.sections:
            self.sections[section_id].is_loaded = True
            self.loaded_sections.add(section_id)
            logger.debug(f"Marked section {section_id} as loaded")

    def mark_unloaded(self, section_id: str):
        """Mark a section as unloaded.

        Args:
            section_id: Section to mark
        """
        if section_id in self.sections:
            self.sections[section_id].is_loaded = False
            self.loaded_sections.discard(section_id)
            logger.debug(f"Marked section {section_id} as unloaded")

    def get_path_sections(
        self, start_section_id: str, goal_section_id: str
    ) -> Optional[List[str]]:
        """Compute path through sections from start to goal.

        Args:
            start_section_id: Starting section
            goal_section_id: Goal section

        Returns:
            List of section IDs forming path, or None if no path exists
        """
        try:
            path = nx.shortest_path(self.graph, start_section_id, goal_section_id)
            return path
        except nx.NetworkXNoPath:
            logger.warning(
                f"No path found from {start_section_id} to {goal_section_id}"
            )
            return None

    @classmethod
    def from_pickle(cls, pickle_path: Path) -> "SceneGraph":
        """Load scene graph from pickle file (similar to NaviSim's sequence_graph).

        Args:
            pickle_path: Path to pickled NetworkX graph

        Returns:
            SceneGraph instance
        """
        import pickle

        logger.info(f"Loading scene graph from: {pickle_path}")

        with open(pickle_path, "rb") as f:
            graph_data = pickle.load(f)

        scene_graph = cls(graph_data)

        # Extract section data from graph nodes
        for node_id, node_data in graph_data.nodes(data=True):
            if "section" in node_data:
                section = node_data["section"]
                scene_graph.sections[node_id] = section
                scene_graph._section_centers.append(section.center)
                scene_graph._section_ids.append(node_id)

        logger.info(
            f"Loaded scene graph with {len(scene_graph.sections)} sections"
        )
        return scene_graph

    def to_pickle(self, pickle_path: Path):
        """Save scene graph to pickle file.

        Args:
            pickle_path: Path to save pickle
        """
        import pickle

        logger.info(f"Saving scene graph to: {pickle_path}")

        with open(pickle_path, "wb") as f:
            pickle.dump(self.graph, f)

    def visualize(self, output_path: Optional[Path] = None):
        """Visualize the scene graph (requires matplotlib).

        Args:
            output_path: Optional path to save visualization
        """
        try:
            import matplotlib.pyplot as plt

            pos = {
                node_id: section.center[:2]  # Use x, y for 2D layout
                for node_id, section in self.sections.items()
            }

            plt.figure(figsize=(12, 8))
            nx.draw(
                self.graph,
                pos,
                with_labels=True,
                node_color="lightblue",
                node_size=500,
                font_size=8,
                font_weight="bold",
            )

            # Highlight loaded sections
            loaded_nodes = list(self.loaded_sections)
            if loaded_nodes:
                nx.draw_networkx_nodes(
                    self.graph,
                    pos,
                    nodelist=loaded_nodes,
                    node_color="lightgreen",
                    node_size=500,
                )

            plt.title("Scene Graph (Green = Loaded)")
            plt.axis("equal")

            if output_path:
                plt.savefig(output_path)
                logger.info(f"Saved visualization to: {output_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib not available, skipping visualization")

    def __len__(self) -> int:
        """Get number of sections."""
        return len(self.sections)

    def __repr__(self) -> str:
        return (
            f"SceneGraph(sections={len(self.sections)}, "
            f"loaded={len(self.loaded_sections)})"
        )
