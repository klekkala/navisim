"""Rendering component for visual representation."""

import numpy as np
from typing import Optional, Dict, Any
from ..entity import Component


class RenderingComponent(Component):
    """
    Rendering component following SAPIEN patterns.
    
    Manages visual representation and rendering properties.
    """
    
    def __init__(
        self,
        visible: bool = True,
        color: np.ndarray = None,
        material_properties: Dict[str, Any] = None
    ):
        """
        Initialize rendering component.
        
        Args:
            visible: Whether the entity is visible
            color: RGB color (0-1 range)
            material_properties: Material properties for rendering
        """
        super().__init__()
        
        self.visible = visible
        self.color = color if color is not None else np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.material_properties = material_properties or {}
        
        # Rendering state
        self.render_data: Optional[np.ndarray] = None
        self.last_render_time = 0.0
    
    def update(self, dt: float) -> None:
        """Update rendering state."""
        # Rendering updates typically handled by rendering system
        pass
    
    def set_visible(self, visible: bool) -> None:
        """Set visibility state."""
        self.visible = visible
    
    def set_color(self, color: np.ndarray) -> None:
        """Set entity color."""
        self.color = np.clip(color, 0.0, 1.0)
    
    def set_material_property(self, property_name: str, value: Any) -> None:
        """Set a material property."""
        self.material_properties[property_name] = value
    
    def get_material_property(self, property_name: str, default: Any = None) -> Any:
        """Get a material property."""
        return self.material_properties.get(property_name, default)
    
    def reset(self) -> None:
        """Reset rendering component."""
        self.render_data = None
        self.last_render_time = 0.0
    
    def __repr__(self) -> str:
        return f"RenderingComponent(visible={self.visible}, color={self.color})"