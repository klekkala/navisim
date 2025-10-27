"""
Boundary Polygon Module

Handles boundary polygon data for spatial constraints in navigation.
"""

from typing import Optional, List, Tuple
import numpy as np


class BoundaryPolygon:
    """
    Boundary polygon data for a sector.

    Defines the spatial boundaries/constraints for navigation within a sector.
    The boundary polygon can be used for:
    - Collision checking
    - Valid navigation area definition
    - Spatial planning constraints
    """

    def __init__(self, seq_id: str, sector_id: str, data_root: Optional[str] = None):
        """
        Initialize BoundaryPolygon.

        Args:
            seq_id: Sequence identifier
            sector_id: Sector identifier
            data_root: Root directory for data files (uses default if None)
        """
        self.seq_id = seq_id
        self.sector_id = sector_id
        self.data_root = data_root or self._get_default_data_root()
        self._vertices: Optional[np.ndarray] = None

    def _get_default_data_root(self) -> str:
        """
        Get default data root path.

        Returns:
            Default data root directory
        """
        # TODO: Configure this through config file
        return "/path/to/data/root"

    def load(self) -> np.ndarray:
        """
        Load boundary polygon data from disk.

        Returns:
            Polygon vertices as numpy array of shape (N, 2) or (N, 3)

        Raises:
            FileNotFoundError: If boundary polygon file not found
        """
        if self._vertices is None:
            # TODO: Implement actual loading logic
            boundary_path = f"{self.data_root}/{self.seq_id}/{self.sector_id}/boundary.npy"
            try:
                self._vertices = np.load(boundary_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Boundary polygon not found at {boundary_path}")
        return self._vertices

    @property
    def vertices(self) -> np.ndarray:
        """
        Get polygon vertices, loading if necessary.

        Returns:
            Polygon vertices
        """
        return self.load()

    @property
    def num_vertices(self) -> int:
        """
        Get number of vertices in the polygon.

        Returns:
            Number of vertices
        """
        if self._vertices is not None:
            return len(self._vertices)
        return 0

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is inside the boundary polygon.

        Args:
            point: (x, y) coordinate to check

        Returns:
            True if point is inside polygon, False otherwise
        """
        vertices = self.load()
        # TODO: Implement point-in-polygon test
        # Can use ray casting algorithm or winding number algorithm
        raise NotImplementedError("contains_point not yet implemented")

    def get_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get axis-aligned bounding box of the polygon.

        Returns:
            ((min_x, min_y), (max_x, max_y))
        """
        vertices = self.load()
        min_coords = np.min(vertices[:, :2], axis=0)
        max_coords = np.max(vertices[:, :2], axis=0)
        return (tuple(min_coords), tuple(max_coords))

    def get_edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get polygon edges as pairs of vertices.

        Returns:
            List of (start_vertex, end_vertex) tuples
        """
        vertices = self.load()
        edges = []
        for i in range(len(vertices)):
            start = vertices[i]
            end = vertices[(i + 1) % len(vertices)]
            edges.append((start, end))
        return edges

    def unload(self) -> None:
        """
        Unload boundary polygon data to free memory.
        """
        self._vertices = None

    def __repr__(self) -> str:
        """String representation."""
        return f"BoundaryPolygon({self.seq_id}/{self.sector_id})"
