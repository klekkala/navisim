"""
Terrain Manager Module

Manages terrain and height field data in IsaacSim.
"""

from typing import Optional, Tuple, Any
import numpy as np


class TerrainManager:
    """
    Manages terrain in IsaacSim environment.

    This class handles:
    - Loading height field data into IsaacSim
    - Terrain mesh generation from USD heightmaps
    - Terrain physics properties (friction, restitution)
    - Collision mesh setup for PhysX
    """

    def __init__(self, stage=None):
        """
        Initialize TerrainManager.

        Args:
            stage: IsaacSim USD stage (from omni.isaac.core)
        """
        self.stage = stage
        self.terrain_loaded = False
        self.current_terrain = None
        self.terrain_prim_path = None

    def load_terrain_from_usd(
        self,
        usd_scene_path: str,
        heightmap_prim_name: str = "Heightmap",
        enable_physics: bool = True
    ) -> str:
        """
        Load terrain from USD scene containing heightmap mesh.

        Args:
            usd_scene_path: Path to USD scene prim (e.g., "/World/sector_usd")
            heightmap_prim_name: Name of heightmap mesh prim
            enable_physics: Whether to enable physics collision

        Returns:
            Prim path of loaded terrain
        """
        heightmap_path = f"{usd_scene_path}/{heightmap_prim_name}"

        # In real IsaacSim implementation:
        # from pxr import UsdGeom, UsdPhysics
        # mesh_prim = self.stage.GetPrimAtPath(heightmap_path)
        #
        # if enable_physics:
        #     # Add collision shape
        #     UsdPhysics.CollisionAPI.Apply(mesh_prim)
        #     UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)

        print(f"Loading terrain from USD: {heightmap_path}")
        print(f"Physics enabled: {enable_physics}")

        self.terrain_prim_path = heightmap_path
        self.terrain_loaded = True
        self.current_terrain = {"path": heightmap_path, "physics": enable_physics}

        return heightmap_path

    def load_height_field(
        self,
        elevation_map: np.ndarray,
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        position: Tuple[float, float, float] = (0, 0, 0)
    ) -> str:
        """
        Load height field from elevation map array.

        Args:
            elevation_map: 2D elevation map data (H, W)
            scale: Scale factor for terrain (x, y, z)
            position: World position for terrain

        Returns:
            Prim path of created terrain
        """
        if elevation_map.ndim != 2:
            raise ValueError(f"Elevation map must be 2D, got shape {elevation_map.shape}")

        height, width = elevation_map.shape

        # In real IsaacSim implementation:
        # from omni.isaac.core.objects import DynamicCuboid
        # from pxr import Gf, UsdGeom
        #
        # Create mesh from elevation data
        # mesh_prim = create_mesh_from_heightmap(elevation_map, scale, position)

        print(f"Creating terrain from elevation map: {elevation_map.shape}")
        print(f"Scale: {scale}, Position: {position}")

        terrain_path = f"/World/Terrain_{id(elevation_map)}"
        self.terrain_prim_path = terrain_path
        self.terrain_loaded = True
        self.current_terrain = {
            "path": terrain_path,
            "shape": elevation_map.shape,
            "scale": scale
        }

        return terrain_path

    def set_terrain_properties(
        self,
        friction: float = 0.8,
        restitution: float = 0.1,
        dynamic_friction: Optional[float] = None
    ) -> None:
        """
        Set physics properties for the terrain.

        Args:
            friction: Static friction coefficient (0.0 - 1.0)
            restitution: Bounciness/restitution coefficient (0.0 - 1.0)
            dynamic_friction: Dynamic friction coefficient (defaults to friction if None)
        """
        if not self.terrain_loaded or self.terrain_prim_path is None:
            print("Warning: No terrain loaded, cannot set properties")
            return

        if dynamic_friction is None:
            dynamic_friction = friction

        # In real IsaacSim implementation:
        # from pxr import UsdPhysics, PhysxSchema
        #
        # terrain_prim = self.stage.GetPrimAtPath(self.terrain_prim_path)
        # physics_material = UsdPhysics.MaterialAPI.Apply(terrain_prim)
        # physics_material.CreateStaticFrictionAttr().Set(friction)
        # physics_material.CreateDynamicFrictionAttr().Set(dynamic_friction)
        # physics_material.CreateRestitutionAttr().Set(restitution)

        print(f"Setting terrain physics properties:")
        print(f"  Static friction: {friction}")
        print(f"  Dynamic friction: {dynamic_friction}")
        print(f"  Restitution: {restitution}")

        if self.current_terrain:
            self.current_terrain.update({
                "friction": friction,
                "dynamic_friction": dynamic_friction,
                "restitution": restitution
            })

    def get_height_at_position(self, x: float, y: float) -> Optional[float]:
        """
        Query terrain height at a specific (x, y) world position.

        Args:
            x: World X coordinate
            y: World Y coordinate

        Returns:
            Height (Z coordinate) at position, or None if not available
        """
        if not self.terrain_loaded:
            return None

        # In real IsaacSim implementation:
        # Use raycasting or mesh query
        # from omni.isaac.core.utils.raycast import raycast
        # hit = raycast(origin=(x, y, 1000), direction=(0, 0, -1))
        # return hit.position[2] if hit else None

        print(f"Querying height at position ({x}, {y})")
        return 0.0  # Placeholder

    def clear_terrain(self) -> None:
        """
        Clear the current terrain from the simulation.
        """
        if self.terrain_prim_path and self.stage:
            # In real IsaacSim:
            # self.stage.RemovePrim(self.terrain_prim_path)
            print(f"Clearing terrain: {self.terrain_prim_path}")

        self.terrain_loaded = False
        self.current_terrain = None
        self.terrain_prim_path = None

    def is_loaded(self) -> bool:
        """
        Check if terrain is currently loaded.

        Returns:
            True if terrain is loaded, False otherwise
        """
        return self.terrain_loaded
