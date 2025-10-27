"""
Sector Module

Represents a sector within a sequence with lazy-loaded spatial data.
"""

from typing import Optional, Any


class Sector:
    """
    Represents a sector in the navigation sequence.

    Each sector contains:
    - Height field data for terrain generation
    - USDZ file for scene representation
    - Boundary polygon for spatial constraints

    Data is lazy-loaded to conserve memory.
    """

    def __init__(self, seq_id: str, sector_id: str):
        """
        Initialize Sector.

        Args:
            seq_id: Sequence identifier
            sector_id: Sector identifier within the sequence
        """
        self.seq_id = seq_id
        self.sector_id = sector_id

        # Lazy-loaded spatial data
        self._height_field = None
        self._usdz = None
        self._boundary = None

        # Links to adjacent sectors
        self.prev: Optional['Sector'] = None
        self.next: Optional['Sector'] = None

    @property
    def height_field(self) -> Any:
        """
        Get height field data for this sector.

        Lazy-loads the height field on first access.

        Returns:
            Height field data
        """
        if self._height_field is None:
            print(f"Loading HeightField for {self.seq_id}/{self.sector_id}")
            from .height_field import HeightField
            self._height_field = HeightField(self.seq_id, self.sector_id)
        return self._height_field

    @property
    def usdz(self) -> Any:
        """
        Get USDZ scene data for this sector.

        Lazy-loads the USDZ data on first access.

        Returns:
            USDZ scene data
        """
        if self._usdz is None:
            print(f"Loading USDZ for {self.seq_id}/{self.sector_id}")
            from .usdz_loader import USDZLoader
            self._usdz = USDZLoader(self.seq_id, self.sector_id)
        return self._usdz

    @property
    def boundary(self) -> Any:
        """
        Get boundary polygon for this sector.

        Lazy-loads the boundary polygon on first access.

        Returns:
            Boundary polygon data
        """
        if self._boundary is None:
            print(f"Loading BoundaryPolygon for {self.seq_id}/{self.sector_id}")
            from .boundary_polygon import BoundaryPolygon
            self._boundary = BoundaryPolygon(self.seq_id, self.sector_id)
        return self._boundary

    def unload_all(self) -> None:
        """
        Unload all lazy-loaded data to free memory.
        """
        self._height_field = None
        self._usdz = None
        self._boundary = None

    def __repr__(self) -> str:
        """String representation of the sector."""
        return f"Sector {self.sector_id}"
