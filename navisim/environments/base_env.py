"""
SAPIEN-style base environment for Navisim.
Follows SAPIEN's SapienEnv pattern with _build_world() and standardized lifecycle.
"""

import uuid
import random
from abc import abstractmethod
from typing import Dict, Tuple, Any, Optional

import gymnasium as gym
import numpy as np
from dataclasses import dataclass, field

from ..core.scene import NavisimScene
from ..simulation.physics.bridge import NavisimBridgeClient
from ..assets.loaders.sequence_graph import SequenceGraph
from ..utils.spaces import RenderMode


@dataclass
class AgentState:
    """Agent state representation following SAPIEN patterns."""
    pose_world_T_agent: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


@dataclass
class RenderFrame:
    """Cached render frame data."""
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    render_time_ms: float = 0.0


class NavisimEnv(gym.Env):
    """
    SAPIEN-style base environment for Navisim.
    
    Follows SAPIEN's architecture patterns:
    - Abstract _build_world() method for subclasses
    - Standardized simulation lifecycle
    - Modular component integration
    - Physics-first design with rendering as component
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        control_freq: int = 60,
        timestep: Optional[float] = None,
        render_mode: str = "rgb_array",
        **kwargs
    ):
        """
        Initialize base environment following SAPIEN patterns.
        
        Args:
            control_freq: Control frequency (acts as frame skip)
            timestep: Physics timestep (computed from control_freq if None)
            render_mode: Rendering mode
        """
        super().__init__()
        
        # Core simulation parameters
        self.control_freq = control_freq
        self.timestep = timestep or (1.0 / control_freq)
        self.render_mode = render_mode
        
        # Core components (initialized in _build_world)
        self._scene: Optional[NavisimScene] = None
        self._bridge: Optional[NavisimBridgeClient] = None
        
        # State management
        self._agent_state = AgentState()
        self._render_frame = RenderFrame()
        self._step_count = 0
        
        # Initialize simulation world
        self._build_world()
        self._setup_spaces()
    
    @abstractmethod
    def _build_world(self) -> None:
        """
        Build the simulation world.
        Subclasses must implement this method following SAPIEN patterns.
        """
        raise NotImplementedError("Subclasses must implement _build_world()")
    
    @abstractmethod
    def _setup_spaces(self) -> None:
        """Setup action and observation spaces."""
        raise NotImplementedError("Subclasses must implement _setup_spaces()")
    
    @abstractmethod
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        raise NotImplementedError("Subclasses must implement _get_observation()")
    
    @abstractmethod
    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        raise NotImplementedError("Subclasses must implement _compute_reward()")
    
    @abstractmethod
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        raise NotImplementedError("Subclasses must implement _check_termination()")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment following SAPIEN patterns."""
        super().reset(seed=seed)
        
        # Reset simulation state
        self._step_count = 0
        self._reset_simulation()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Step environment following SAPIEN patterns."""
        # Apply action through bridge
        self._apply_action(action)
        
        # Step physics simulation
        self._step_simulation()
        
        # Update agent state
        self._update_agent_state()
        
        # Compute reward and termination
        observation = self._get_observation()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = False  # Can be overridden by subclasses
        info = self._get_info()
        
        self._step_count += 1
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render environment."""
        if self._render_frame.rgb is not None:
            return self._render_frame.rgb
        return None
    
    def close(self):
        """Close environment and cleanup resources."""
        if self._bridge:
            self._bridge.close()
    
    def seed(self, seed: Optional[int] = None) -> list:
        """Seed environment following SAPIEN patterns."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return [seed]
    
    # ==================
    # Internal Methods
    # ==================
    
    def _reset_simulation(self) -> None:
        """Reset simulation state (can be overridden by subclasses)."""
        pass
    
    def _apply_action(self, action) -> None:
        """Apply action to simulation (should be overridden by subclasses)."""
        pass
    
    def _step_simulation(self) -> None:
        """Step physics simulation (should be overridden by subclasses)."""
        pass
    
    def _update_agent_state(self) -> None:
        """Update agent state from simulation (should be overridden by subclasses)."""
        pass
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        return {
            "step_count": self._step_count,
            "timestep": self.timestep,
            "control_freq": self.control_freq,
        }
    
    # ==================
    # Properties
    # ==================
    
    @property
    def scene(self) -> Optional[NavisimScene]:
        """Get current scene."""
        return self._scene
    
    @property
    def agent_state(self) -> AgentState:
        """Get current agent state."""
        return self._agent_state
    
    @property
    def step_count(self) -> int:
        """Get current step count."""
        return self._step_count