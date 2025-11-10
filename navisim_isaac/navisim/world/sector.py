"""
Sector Module

Represents a navigation sector with USD scene data.
"""

from typing import Optional, Any, Dict
import numpy as np


class Sector:
    """
    Represents a sector in the navigation environment.

    Each sector is defined by a single USD file containing:
    - Gaussian splat scene (for RL visual input)
    - Height field mesh (for physics/collision detection)
    - Optional boundary polygon (for spatial constraints)

    Data is lazy-loaded to conserve memory.
    """

    def __init__(
        self,
        sector_id: str,
        usd_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Sector.

        Args:
            sector_id: Sector identifier (node ID in graph)
            usd_path: Path to USD file containing all sector data
            metadata: Optional metadata from graph node
        """
        self.sector_id = sector_id
        self.usd_path = usd_path
        self.metadata = metadata or {}

        # Lazy-loaded data extracted from USD
        self._height_field = None
        self._boundary = None
        self._usd_stage = None

    @property
    def scene_path(self) -> str:
        """
        Get USD scene path for loading into IsaacSim.

        Returns:
            Path to USD file
        """
        return self.usd_path

    @property
    def height_field(self) -> np.ndarray:
        """
        Get height field data extracted from USD file.

        Lazy-loads the height field mesh on first access.
        The height field is used for physics/collision detection.

        Returns:
            Height field data as numpy array

        Raises:
            FileNotFoundError: If USD file not found
            ValueError: If no heightmap found in USD
        """
        if self._height_field is None:
            print(f"Extracting height field from USD: {self.sector_id}")
            from ..spaces.usd_extractors import extract_heightmap_from_usd
            self._height_field = extract_heightmap_from_usd(self.usd_path)
        return self._height_field

    @property
    def boundary(self) -> Optional[np.ndarray]:
        """
        Get boundary polygon extracted from USD file.

        Lazy-loads the boundary polygon on first access.
        Boundary is optional and may not exist in all sectors.

        Returns:
            Boundary polygon vertices as numpy array, or None if not present
        """
        if self._boundary is None:
            print(f"Extracting boundary from USD: {self.sector_id}")
            from ..spaces.usd_extractors import extract_boundary_from_usd
            try:
                self._boundary = extract_boundary_from_usd(self.usd_path)
            except ValueError:
                # Boundary is optional
                self._boundary = None
        return self._boundary

    @property
    def gaussian_scene_path(self) -> str:
        """
        Get path to Gaussian splat scene within USD.

        The Gaussian scene is used for RL visual input.

        Returns:
            Prim path to Gaussian scene within USD (e.g., "/World/GaussianScene")
        """
        # Gaussian scene is embedded in the USD file
        # IsaacSim will render it when loading the USD
        return f"{self.usd_path}:/World/GaussianScene"

    @property
    def heightmap_prim_path(self) -> str:
        """
        Get prim path to heightmap mesh within USD.

        Returns:
            Prim path to heightmap (e.g., "/World/Heightmap")
        """
        return "/World/Heightmap"

    def has_usd_file(self) -> bool:
        """
        Check if USD file exists.

        Returns:
            True if USD file exists, False otherwise
        """
        import os
        return os.path.exists(self.usd_path)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata value from graph node data.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)

    def unload_all(self) -> None:
        """
        Unload all lazy-loaded data to free memory.
        """
        self._height_field = None
        self._boundary = None
        self._usd_stage = None

    def __repr__(self) -> str:
        """String representation of the sector."""
        return f"Sector({self.sector_id}, usd={self.usd_path})"
