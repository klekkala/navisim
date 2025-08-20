"""
Navisim: High-fidelity simulation toolkit for autonomous navigation.

Follows SAPIEN architecture patterns with entity-component system,
modular design, and standardized environment interfaces.
"""

__version__ = "0.2.0"

# Core SAPIEN-style components
from .core.scene import NavisimScene
from .core.entity import Entity
from .core.components import (
    TransformComponent,
    PhysicsComponent,
    RenderingComponent,
    SensorComponent
)

# SAPIEN-style environments
from .environments.base_env import NavisimEnv
from .environments.navigation_env import NavigationEnv

# Asset management
from .assets.loaders.sequence_graph import SequenceGraph
from .assets.loaders.sector import Sector

# Simulation backends
from .simulation.physics.bridge import NavisimBridgeClient

# Utilities
from .utils.spaces import RenderMode
from .utils.transforms import pose_to_matrix, matrix_to_pose

# Legacy compatibility (deprecated)
from .envs.navisim_env import NavisimEnv as LegacyNavisimEnv

__all__ = [
    # Core components
    "NavisimScene",
    "Entity", 
    "TransformComponent",
    "PhysicsComponent",
    "RenderingComponent", 
    "SensorComponent",
    
    # Environments
    "NavisimEnv",
    "NavigationEnv",
    
    # Assets
    "SequenceGraph",
    "Sector",
    
    # Simulation
    "NavisimBridgeClient",
    
    # Utilities
    "RenderMode",
    "pose_to_matrix",
    "matrix_to_pose",
    
    # Legacy (deprecated)
    "LegacyNavisimEnv",
]