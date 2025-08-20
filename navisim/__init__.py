"""
Navisim: High-fidelity simulation toolkit for autonomous navigation.

Follows ManiSkill architecture patterns with task-oriented design,
hybrid Isaac Sim + Gaussian Splatting simulation, and standardized environment interfaces.
"""

__version__ = "0.3.0"

# ManiSkill-style environments
from .envs.base import BaseEnv, register_env, make_env
from .envs.tasks.navigation import NavigationTask

# ManiSkill-style agents
from .agents.base_agent import BaseAgent
from .agents.navigation_agent import NavigationAgent

# Core entity-component system (legacy compatibility)
from .core.scene import NavisimScene
from .core.entity import Entity
from .core.components import (
    TransformComponent,
    PhysicsComponent,
    RenderingComponent,
    SensorComponent
)

# Asset management
from .assets.loaders.sequence_graph import SequenceGraph
from .assets.loaders.sector import Sector

# Simulation backends
from .simulation.physics.bridge import NavisimBridgeClient
from .simulation.physics.isaac_sim import IsaacSimBackend
from .simulation.rendering.camera import NavisimCamera

# Utilities
from .utils.spaces import RenderMode, RlPolicy, RelativeDir
from .utils.transforms import pose_to_matrix, matrix_to_pose

# Legacy compatibility - these directories have been removed in cleanup

__all__ = [
    # ManiSkill-style environments
    "BaseEnv",
    "register_env", 
    "make_env",
    "NavigationTask",
    
    # ManiSkill-style agents
    "BaseAgent",
    "NavigationAgent",
    
    # Core components (legacy)
    "NavisimScene",
    "Entity", 
    "TransformComponent",
    "PhysicsComponent",
    "RenderingComponent", 
    "SensorComponent",
    
    # Assets
    "SequenceGraph",
    "Sector",
    
    # Simulation
    "NavisimBridgeClient",
    "IsaacSimBackend",
    "NavisimCamera",
    
    # Utilities
    "RenderMode",
    "RlPolicy", 
    "RelativeDir",
    "pose_to_matrix",
    "matrix_to_pose",
    
    # Legacy components removed during cleanup
]