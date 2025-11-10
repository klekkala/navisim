"""
IsaacSim Simulator Module

Main interface for IsaacSim 5.0 simulation environment.

IMPORTANT: SimulationApp must be initialized before importing other Isaac Sim modules.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

# Global SimulationApp instance
_simulation_app = None

def get_simulation_app():
    """Get the global SimulationApp instance."""
    return _simulation_app


class IsaacSimulator:
    """
    Main IsaacSim 5.0 simulator interface.

    This class handles:
    - IsaacSim SimulationApp and World initialization
    - Scene loading from USD files
    - Robot spawning and management
    - Physics simulation stepping
    - Sensor data collection

    IMPORTANT: This must be created before importing other Isaac Sim modules.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize IsaacSim simulator.

        Args:
            config: Configuration dictionary for the simulator
        """
        self.config = config or {}
        self.is_running = False
        self.world = None
        self.stage = None
        self.simulation_app = None

        # Simulation parameters
        self.physics_dt = self.config.get('physics_dt', 1.0 / 60.0)  # 60 Hz
        self.rendering_dt = self.config.get('rendering_dt', 1.0 / 30.0)  # 30 Hz

    def initialize(self, headless: bool = False) -> None:
        """
        Initialize the IsaacSim SimulationApp, World and stage.

        Args:
            headless: Whether to run without GUI
        """
        global _simulation_app

        # Step 1: Initialize SimulationApp FIRST (before any other imports)
        try:
            from isaacsim import SimulationApp
        except ImportError:
            print("Warning: IsaacSim not available. Running in simulation mode.")
            self.is_running = True
            return

        print("Initializing IsaacSim SimulationApp...")

        # Create SimulationApp with configuration
        simulation_config = {
            "headless": headless,
            "width": self.config.get('window_width', 1280),
            "height": self.config.get('window_height', 720),
        }

        self.simulation_app = SimulationApp(simulation_config)
        _simulation_app = self.simulation_app

        print("✓ SimulationApp created")

        # Step 2: NOW import Isaac Sim modules (after SimulationApp is created)
        try:
            from omni.isaac.core import World
            from omni.isaac.core.utils.stage import get_current_stage
            from pxr import Usd, UsdGeom, UsdPhysics
        except ImportError as e:
            print(f"Error importing Isaac Sim modules: {e}")
            self.is_running = True
            return

        # Step 3: Create World instance
        print("Initializing IsaacSim World...")
        self.world = World(
            physics_dt=self.physics_dt,
            rendering_dt=self.rendering_dt,
            stage_units_in_meters=1.0
        )

        # Step 4: Get USD stage
        self.stage = get_current_stage()

        # Step 5: Set up scene
        self._setup_scene()

        self.is_running = True
        print("✓ IsaacSim World initialized successfully")

    def _setup_scene(self) -> None:
        """
        Set up the basic scene (lighting, ground plane, etc.).
        """
        if self.world is None:
            return

        try:
            # Add default ground plane (can be replaced with terrain later)
            from omni.isaac.core.objects import GroundPlane
            GroundPlane(
                prim_path="/World/defaultGroundPlane",
                size=100.0,
                color=np.array([0.5, 0.5, 0.5])
            )

            # Add default lighting
            from omni.isaac.core.utils.prims import create_prim
            create_prim(
                "/World/defaultLight",
                "DistantLight",
                attributes={"intensity": 500}
            )
        except Exception as e:
            print(f"Warning: Could not setup default scene: {e}")

    def load_sector_usd(
        self,
        usd_path: str,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> str:
        """
        Load a sector USD file into the scene.

        Args:
            usd_path: Path to sector USD file
            position: World position offset

        Returns:
            Prim path of loaded scene
        """
        if self.stage is None:
            print(f"Would load USD: {usd_path} at {position}")
            return "/World/Sector"

        try:
            from omni.isaac.core.utils.stage import add_reference_to_stage
            from pxr import UsdGeom

            prim_path = "/World/Sector"
            add_reference_to_stage(usd_path, prim_path)

            # Set position
            xform = UsdGeom.Xformable(self.stage.GetPrimAtPath(prim_path))
            xform.AddTranslateOp().Set(position)

            print(f"Loaded sector USD: {usd_path}")
            return prim_path
        except Exception as e:
            print(f"Error loading USD: {e}")
            return "/World/Sector"

    def step(self, num_steps: int = 1) -> None:
        """
        Step the simulation forward.

        Args:
            num_steps: Number of physics steps to simulate
        """
        if self.world is None:
            return

        for _ in range(num_steps):
            self.world.step(render=True)

    def reset(self) -> None:
        """
        Reset the simulation world.
        """
        if self.world is None:
            print("Resetting simulation")
            return

        print("Resetting IsaacSim World...")
        self.world.reset()

    def play(self) -> None:
        """
        Start physics simulation.
        """
        if self.world is None:
            print("Starting simulation")
            return

        self.world.play()
        print("Simulation started")

    def pause(self) -> None:
        """
        Pause physics simulation.
        """
        if self.world is None:
            return

        self.world.pause()
        print("Simulation paused")

    def stop(self) -> None:
        """
        Stop physics simulation.
        """
        if self.world is None:
            return

        self.world.stop()
        print("Simulation stopped")

    def get_stage(self):
        """
        Get the USD stage.

        Returns:
            USD stage or None
        """
        return self.stage

    def get_world(self):
        """
        Get the World instance.

        Returns:
            World instance or None
        """
        return self.world

    def is_playing(self) -> bool:
        """
        Check if simulation is playing.

        Returns:
            True if simulation is running
        """
        if self.world is None:
            return self.is_running

        return self.world.is_playing()

    def shutdown(self) -> None:
        """
        Shutdown the IsaacSim environment.
        """
        if self.world is not None:
            self.world.stop()
            self.world.clear()

        if self.simulation_app is not None:
            self.simulation_app.close()

        self.is_running = False
        print("IsaacSim shutdown complete")
