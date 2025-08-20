"""ManiSkill-style BaseEnv for Navisim."""

import gym
import numpy as np
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..simulation.physics.isaac_sim import IsaacSimBackend


@dataclass
class SceneConfig:
    """Configuration for scene setup."""
    sim_backend: str = "isaac_sim"
    rendering_backend: str = "gaussian_splatting"


class BaseEnv(gym.Env):
    """
    ManiSkill-style base environment for Navisim.
    
    Follows ManiSkill patterns:
    - Standardized observation/action spaces
    - Task-oriented design
    - Environment registration system
    - Hybrid Isaac Sim + Gaussian Splatting rendering
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        obs_mode: str = "state_dict",
        render_mode: Optional[str] = None,
        sim_backend: str = "isaac_sim",
        **kwargs
    ):
        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self.sim_backend = sim_backend
        
        # Scene configuration
        self.scene_config = SceneConfig(
            sim_backend=sim_backend,
            **kwargs
        )
        
        # Initialize simulation backend
        self._init_simulation_backend()
        
        # Build the world (ManiSkill pattern)
        self._build_world()
        
        # Setup observation and action spaces
        self._setup_spaces()
        
        # Initialize state
        self._episode_step = 0
        self._max_episode_steps = kwargs.get("max_episode_steps", 1000)
        
    def _init_simulation_backend(self):
        """Initialize the simulation backend."""
        if self.sim_backend == "isaac_sim":
            self.sim = IsaacSimBackend()
        else:
            raise ValueError(f"Unknown simulation backend: {self.sim_backend}")
    
    @abstractmethod
    def _build_world(self):
        """Build the simulation world. Override in subclasses."""
        pass
    
    @abstractmethod
    def _setup_spaces(self):
        """Setup observation and action spaces. Override in subclasses."""
        pass
    
    @abstractmethod
    def _get_obs(self) -> Dict[str, Any]:
        """Get current observations. Override in subclasses."""
        pass
    
    @abstractmethod
    def _get_info(self) -> Dict[str, Any]:
        """Get info dict. Override in subclasses."""
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset simulation
        self._reset_simulation()
        
        # Reset episode state
        self._episode_step = 0
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """Step the environment."""
        # Apply action
        self._apply_action(action)
        
        # Step simulation
        self._step_simulation()
        
        # Get observations and rewards
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        info = self._get_info()
        
        self._episode_step += 1
        
        return obs, reward, terminated, truncated, info
    
    @abstractmethod
    def _reset_simulation(self):
        """Reset the simulation state."""
        pass
    
    @abstractmethod
    def _apply_action(self, action):
        """Apply action to the simulation."""
        pass
    
    @abstractmethod
    def _step_simulation(self):
        """Step the simulation forward."""
        pass
    
    @abstractmethod
    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        pass
    
    def _check_terminated(self) -> bool:
        """Check if episode is terminated."""
        return False
    
    def _check_truncated(self) -> bool:
        """Check if episode is truncated."""
        return self._episode_step >= self._max_episode_steps
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            # Display to screen
            self._render_human()
        elif self.render_mode == "rgb_array":
            # Return RGB array
            return self._render_rgb_array()
        else:
            return None
    
    def _render_human(self):
        """Render to human display."""
        pass
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render to RGB array."""
        # Use NavisimCamera for rendering
        if hasattr(self, 'camera'):
            return self.camera.get_rgb()
        else:
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up environment."""
        if hasattr(self, 'sim'):
            self.sim.close()


# Environment registration system (ManiSkill style)
REGISTERED_ENVS = {}

def register_env(uid: str, max_episode_steps: int = 1000, **kwargs):
    """Register environment decorator."""
    def _register(cls):
        if uid in REGISTERED_ENVS:
            print(f"Warning: Environment {uid} already registered, overwriting")
        
        REGISTERED_ENVS[uid] = {
            "cls": cls,
            "max_episode_steps": max_episode_steps,
            **kwargs
        }
        return cls
    return _register


def make_env(env_id: str, **kwargs):
    """Create environment instance."""
    if env_id not in REGISTERED_ENVS:
        raise ValueError(f"Environment {env_id} not registered. Available: {list(REGISTERED_ENVS.keys())}")
    
    env_info = REGISTERED_ENVS[env_id]
    env_cls = env_info["cls"]
    
    # Merge default and provided kwargs
    env_kwargs = {k: v for k, v in env_info.items() if k != "cls"}
    env_kwargs.update(kwargs)
    
    return env_cls(**env_kwargs)