"""
Entity-Component System implementation following SAPIEN patterns.
"""

import uuid
from typing import Dict, Any, Optional, Type
from abc import ABC, abstractmethod


class Component(ABC):
    """Base component class following SAPIEN patterns."""
    
    def __init__(self):
        self.entity: Optional['Entity'] = None
    
    @abstractmethod
    def update(self, dt: float) -> None:
        """Update component state."""
        pass
    
    def reset(self) -> None:
        """Reset component to initial state."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup component resources."""
        pass


class Entity:
    """
    Entity class following SAPIEN's entity-component system.
    
    Entities are containers for components that define behavior and data.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize entity.
        
        Args:
            name: Optional name for the entity
        """
        self.entity_id = str(uuid.uuid4())
        self.name = name or f"Entity_{self.entity_id[:8]}"
        
        self._components: Dict[Type[Component], Component] = {}
        self._scene: Optional['NavisimScene'] = None
    
    def add_component(self, component: Component) -> None:
        """Add a component to this entity."""
        component_type = type(component)
        
        if component_type in self._components:
            raise ValueError(f"Entity already has component of type {component_type}")
        
        self._components[component_type] = component
        component.entity = self
    
    def remove_component(self, component_type: Type[Component]) -> None:
        """Remove a component from this entity."""
        if component_type in self._components:
            component = self._components[component_type]
            component.cleanup()
            component.entity = None
            del self._components[component_type]
    
    def get_component(self, component_type: Type[Component]) -> Optional[Component]:
        """Get a component of the specified type."""
        return self._components.get(component_type)
    
    def has_component(self, component_type: Type[Component]) -> bool:
        """Check if entity has a component of the specified type."""
        return component_type in self._components
    
    def step(self, dt: float) -> None:
        """Update all components."""
        for component in self._components.values():
            component.update(dt)
    
    def reset(self) -> None:
        """Reset all components."""
        for component in self._components.values():
            component.reset()
    
    def cleanup(self) -> None:
        """Cleanup all components."""
        for component in self._components.values():
            component.cleanup()
        self._components.clear()
    
    @property
    def scene(self) -> Optional['NavisimScene']:
        """Get the scene this entity belongs to."""
        return self._scene
    
    @property
    def components(self) -> Dict[Type[Component], Component]:
        """Get all components attached to this entity."""
        return self._components.copy()
    
    def __repr__(self) -> str:
        return f"Entity(id={self.entity_id[:8]}, name={self.name}, components={len(self._components)})"