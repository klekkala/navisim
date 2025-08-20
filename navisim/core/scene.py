"""
SAPIEN-style scene management for Navisim.
Manages entities, components, and simulation state.
"""

import uuid
from typing import Dict, List, Optional, Any
import numpy as np

from .entity import Entity
from ..assets.loaders.sector import Sector
from ..simulation.rendering.camera import NavisimCamera
from ..utils.transforms import pose_to_matrix


class NavisimScene:
    """
    SAPIEN-style scene management.
    
    Manages entities in an entity-component system following SAPIEN patterns.
    Handles scene lifecycle, entity management, and simulation coordination.
    """
    
    def __init__(self, timestep: float = 1.0 / 60.0):
        """
        Initialize scene.
        
        Args:
            timestep: Physics timestep for simulation
        """
        self.timestep = timestep
        self.scene_id = str(uuid.uuid4())
        
        # Entity management
        self._entities: Dict[str, Entity] = {}
        self._active_sector: Optional[Sector] = None
        
        # Scene state
        self._step_count = 0
        self._is_initialized = False
    
    def add_entity(self, entity: Entity) -> None:
        """Add entity to scene."""
        self._entities[entity.entity_id] = entity
        entity._scene = self
    
    def remove_entity(self, entity_id: str) -> None:
        """Remove entity from scene."""
        if entity_id in self._entities:
            entity = self._entities[entity_id]
            entity._scene = None
            del self._entities[entity_id]
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)
    
    def get_entities_with_component(self, component_type: type) -> List[Entity]:
        """Get all entities that have a specific component type."""
        return [
            entity for entity in self._entities.values()
            if entity.has_component(component_type)
        ]
    
    def set_active_sector(self, sector: Sector) -> None:
        """Set the active sector for the scene."""
        if self._active_sector:
            self._active_sector.unload_all()
        
        self._active_sector = sector
        self._active_sector.load()
    
    def get_random_spawn_pose(self) -> np.ndarray:
        """
        Get a random spawn pose within the active sector.
        
        Returns:
            4x4 transformation matrix representing spawn pose
        """
        if not self._active_sector:
            # Default spawn at origin
            return np.eye(4, dtype=np.float64)
        
        # Use sector's spawn logic
        pose_6dof = self._active_sector.random_spawn_pose()
        return pose_to_matrix(pose_6dof)
    
    def get_target_locations(self) -> np.ndarray:
        """
        Get target locations from active sector.
        
        Returns:
            Nx2 array of target positions
        """
        if not self._active_sector:
            return np.zeros((0, 2), dtype=np.float32)
        
        return self._active_sector.get_target_locations()
    
    def step(self) -> None:
        """Step the scene simulation."""
        self._step_count += 1
        
        # Update all entities with physics components
        for entity in self.get_entities_with_component("PhysicsComponent"):
            entity.step(self.timestep)
    
    def reset(self) -> None:
        """Reset scene state."""
        self._step_count = 0
        
        # Reset all entities
        for entity in self._entities.values():
            entity.reset()
    
    def cleanup(self) -> None:
        """Cleanup scene resources."""
        for entity in self._entities.values():
            entity.cleanup()
        
        if self._active_sector:
            self._active_sector.unload_all()
        
        self._entities.clear()
    
    @property
    def entities(self) -> Dict[str, Entity]:
        """Get all entities in scene."""
        return self._entities.copy()
    
    @property
    def active_sector(self) -> Optional[Sector]:
        """Get active sector."""
        return self._active_sector
    
    @property
    def step_count(self) -> int:
        """Get current step count."""
        return self._step_count
    
    @property
    def entity_count(self) -> int:
        """Get number of entities in scene."""
        return len(self._entities)
    
    def __repr__(self) -> str:
        return f"NavisimScene(id={self.scene_id[:8]}, entities={self.entity_count}, steps={self.step_count})"