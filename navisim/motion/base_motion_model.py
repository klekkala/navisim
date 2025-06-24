from abc import ABC, abstractmethod
from typing import Any, Tuple
from navisim.world.sector import Sector

class MotionModel(ABC):
    """
    Abstract base class for motion models used in Navisim.
    Initialized with sector-specific data such as elevation, occupancy, and boundary.
    """
    def __init__(self, sector: Sector):
        self.sector = sector

    @abstractmethod
    def move(self, current_pose: Tuple[float, ...], action: Any) -> Tuple[float, ...]:
        """
        Computes the next pose given the current pose and an action.

        Args:
            current_pose (tuple): The agent's current pose, e.g., (x, y, z, yaw, pitch, roll).
            action (Any): The action provided by the agent (could be velocity, direction, etc.).

        Returns:
            tuple: The updated pose after applying the action.
        """
        pass