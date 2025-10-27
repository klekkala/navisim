"""
Spaces Module

Contains spatial data structures for navigation (height fields, USDZ, boundaries, elevation maps).
"""

from .height_field import HeightField
from .usdz_loader import USDZLoader
from .boundary_polygon import BoundaryPolygon
from .elevation_map import ElevationMapGenerator

__all__ = ["HeightField", "USDZLoader", "BoundaryPolygon", "ElevationMapGenerator"]
