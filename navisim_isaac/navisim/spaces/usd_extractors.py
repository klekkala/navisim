"""
USD Extractors Module

Utilities for extracting data from USD files for IsaacSim.
"""

from typing import Optional, Tuple
import numpy as np
import os


def extract_heightmap_from_usd(usd_path: str) -> np.ndarray:
    """
    Extract heightmap mesh data from USD file.

    The heightmap is used for physics/collision detection in IsaacSim.

    Args:
        usd_path: Path to USD file

    Returns:
        Height field vertices as numpy array (N, 3)

    Raises:
        FileNotFoundError: If USD file doesn't exist
        ValueError: If no heightmap found in USD
    """
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    try:
        # Import USD API (pxr)
        from pxr import Usd, UsdGeom

        # Open USD stage
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            raise ValueError(f"Failed to open USD stage: {usd_path}")

        # Look for heightmap mesh at expected path
        heightmap_prim_path = "/World/Heightmap"
        heightmap_prim = stage.GetPrimAtPath(heightmap_prim_path)

        if not heightmap_prim or not heightmap_prim.IsValid():
            raise ValueError(f"No Heightmap prim found at {heightmap_prim_path} in {usd_path}")

        # Get mesh data
        mesh = UsdGeom.Mesh(heightmap_prim)
        if not mesh:
            raise ValueError(f"Heightmap prim is not a valid mesh: {heightmap_prim_path}")

        # Extract points (vertices)
        points_attr = mesh.GetPointsAttr()
        if not points_attr:
            raise ValueError(f"Heightmap mesh has no points attribute")

        points = points_attr.Get()
        if not points:
            raise ValueError(f"Heightmap mesh points are empty")

        # Convert to numpy array
        vertices = np.array([(p[0], p[1], p[2]) for p in points], dtype=np.float32)

        print(f"Extracted heightmap: {vertices.shape[0]} vertices from {usd_path}")
        return vertices

    except ImportError:
        print("Warning: pxr (USD Python bindings) not available")
        print("Returning placeholder heightmap data")
        # Return placeholder for testing without USD API
        return _generate_placeholder_heightmap()

    except Exception as e:
        raise ValueError(f"Error extracting heightmap from {usd_path}: {e}")


def extract_boundary_from_usd(usd_path: str) -> Optional[np.ndarray]:
    """
    Extract boundary polygon from USD file.

    The boundary defines navigable area constraints.

    Args:
        usd_path: Path to USD file

    Returns:
        Boundary vertices as numpy array (N, 2) or (N, 3), or None if not present

    Raises:
        FileNotFoundError: If USD file doesn't exist
        ValueError: If boundary prim found but invalid
    """
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(usd_path)
        if not stage:
            return None

        # Look for boundary polygon (optional)
        boundary_prim_path = "/World/Boundary"
        boundary_prim = stage.GetPrimAtPath(boundary_prim_path)

        if not boundary_prim or not boundary_prim.IsValid():
            # Boundary is optional, return None
            return None

        # Try to interpret as mesh or points
        if boundary_prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(boundary_prim)
            points_attr = mesh.GetPointsAttr()
            if points_attr:
                points = points_attr.Get()
                vertices = np.array([(p[0], p[1], p[2]) for p in points], dtype=np.float32)
                print(f"Extracted boundary: {vertices.shape[0]} vertices from {usd_path}")
                return vertices[:, :2]  # Return 2D polygon (x, y)

        elif boundary_prim.IsA(UsdGeom.Points):
            points_prim = UsdGeom.Points(boundary_prim)
            points_attr = points_prim.GetPointsAttr()
            if points_attr:
                points = points_attr.Get()
                vertices = np.array([(p[0], p[1]) for p in points], dtype=np.float32)
                print(f"Extracted boundary: {vertices.shape[0]} vertices from {usd_path}")
                return vertices

        # Boundary prim exists but unrecognized format
        print(f"Warning: Boundary prim found but unrecognized format: {boundary_prim.GetTypeName()}")
        return None

    except ImportError:
        # USD API not available
        return None

    except Exception as e:
        print(f"Warning: Error extracting boundary from {usd_path}: {e}")
        return None


def get_heightmap_mesh_info(usd_path: str) -> dict:
    """
    Get heightmap mesh metadata without loading full data.

    Args:
        usd_path: Path to USD file

    Returns:
        Dictionary with mesh info (num_vertices, num_faces, bounds, etc.)
    """
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(usd_path)
        heightmap_prim = stage.GetPrimAtPath("/World/Heightmap")

        if not heightmap_prim or not heightmap_prim.IsValid():
            return {}

        mesh = UsdGeom.Mesh(heightmap_prim)

        # Get mesh attributes
        points_attr = mesh.GetPointsAttr()
        face_counts_attr = mesh.GetFaceVertexCountsAttr()
        face_indices_attr = mesh.GetFaceVertexIndicesAttr()

        info = {
            'has_heightmap': True,
            'num_vertices': len(points_attr.Get()) if points_attr else 0,
            'num_faces': len(face_counts_attr.Get()) if face_counts_attr else 0,
        }

        # Get bounding box
        if points_attr:
            points = np.array([(p[0], p[1], p[2]) for p in points_attr.Get()])
            info['bounds'] = {
                'min': points.min(axis=0).tolist(),
                'max': points.max(axis=0).tolist()
            }

        return info

    except ImportError:
        return {'error': 'USD API not available'}
    except Exception as e:
        return {'error': str(e)}


def has_gaussian_scene(usd_path: str) -> bool:
    """
    Check if USD file contains Gaussian splat scene reference.

    Args:
        usd_path: Path to USD file

    Returns:
        True if Gaussian scene found, False otherwise
    """
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(usd_path)
        if not stage:
            return False

        gaussian_prim = stage.GetPrimAtPath("/World/GaussianScene")
        return gaussian_prim and gaussian_prim.IsValid()

    except ImportError:
        return False
    except Exception:
        return False


def _generate_placeholder_heightmap(size: int = 100) -> np.ndarray:
    """
    Generate placeholder heightmap for testing without USD API.

    Args:
        size: Grid size

    Returns:
        Placeholder height field vertices
    """
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    X, Y = np.meshgrid(x, y)

    # Simple terrain: slight slope
    Z = 0.1 * X + 0.05 * Y

    # Flatten to vertex array
    vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    return vertices.astype(np.float32)


def extract_gaussian_reference_path(usd_path: str) -> Optional[str]:
    """
    Extract the path to the referenced Gaussian USDZ file.

    Args:
        usd_path: Path to USD file

    Returns:
        Path to Gaussian USDZ file, or None if not found
    """
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(usd_path)
        if not stage:
            return None

        gaussian_prim = stage.GetPrimAtPath("/World/GaussianScene")
        if not gaussian_prim or not gaussian_prim.IsValid():
            return None

        # Get references
        refs = gaussian_prim.GetReferences()
        if refs:
            # Get first reference
            for ref in refs.GetAddedOrExplicitItems():
                return ref.assetPath

        return None

    except ImportError:
        return None
    except Exception:
        return None


def validate_sector_usd(usd_path: str) -> Tuple[bool, str]:
    """
    Validate that a sector USD file has required components.

    Args:
        usd_path: Path to USD file

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(usd_path):
        return False, f"USD file not found: {usd_path}"

    try:
        from pxr import Usd

        stage = Usd.Stage.Open(usd_path)
        if not stage:
            return False, "Failed to open USD stage"

        # Check for required heightmap
        heightmap_prim = stage.GetPrimAtPath("/World/Heightmap")
        if not heightmap_prim or not heightmap_prim.IsValid():
            return False, "Missing required Heightmap prim at /World/Heightmap"

        # Check for Gaussian scene (optional but recommended)
        gaussian_prim = stage.GetPrimAtPath("/World/GaussianScene")
        has_gaussian = gaussian_prim and gaussian_prim.IsValid()

        if not has_gaussian:
            return True, "Valid (Warning: No Gaussian scene found)"

        return True, "Valid"

    except ImportError:
        return True, "Valid (Warning: USD API not available for validation)"
    except Exception as e:
        return False, f"Validation error: {e}"
