"""Isaac Sim physics backend implementation."""

import numpy as np
from typing import Dict, Any, Optional
from .bridge import NavisimBridgeClient


class IsaacSimBackend:
    """
    Isaac Sim physics backend following SAPIEN patterns.
    
    Manages physics simulation through Isaac Sim via ZMQ bridge.
    """
    
    def __init__(self, bridge_url: str = "ipc:///tmp/navisim/bridge"):
        """
        Initialize Isaac Sim backend.
        
        Args:
            bridge_url: ZMQ URL for bridge communication
        """
        self.bridge = NavisimBridgeClient(bridge_url)
        self._is_connected = False
    
    def connect(self) -> bool:
        """Connect to Isaac Sim bridge."""
        try:
            # Test connection with a simple control command
            response = self.bridge.control("ping")
            self._is_connected = response.get("ok", False)
            return self._is_connected
        except Exception:
            self._is_connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Isaac Sim bridge."""
        self.bridge.close()
        self._is_connected = False
    
    def step_physics(
        self,
        action: Dict[str, Any],
        dt: float,
        camera_params: Dict[str, Any],
        sector_id: str,
        version: str = "v0"
    ) -> tuple:
        """
        Step physics simulation and get render output.
        
        Args:
            action: Action command (velocity/force)
            dt: Physics timestep
            camera_params: Camera parameters for rendering
            sector_id: Current sector ID
            version: Sector version
            
        Returns:
            Tuple of (header, rgb_image, depth_image)
        """
        if not self._is_connected:
            raise RuntimeError("Isaac Sim backend not connected")
        
        return self.bridge.step_render(
            action=action,
            dt=dt,
            camera=camera_params,
            sector_id=sector_id,
            version=version
        )
    
    def load_sector(self, sector_id: str, version: str = "v0") -> bool:
        """Load a sector in Isaac Sim."""
        try:
            response = self.bridge.control(
                "load_sector",
                sector_id=sector_id,
                version=version
            )
            return response.get("ok", False)
        except Exception:
            return False
    
    def activate_sector(self, sector_id: str, version: str = "v0") -> bool:
        """Activate a sector in Isaac Sim."""
        try:
            response = self.bridge.control(
                "activate_sector",
                sector_id=sector_id,
                version=version
            )
            return response.get("ok", False)
        except Exception:
            return False
    
    @property
    def is_connected(self) -> bool:
        """Check if backend is connected."""
        return self._is_connected