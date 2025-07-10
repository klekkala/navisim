from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch

from gymnasium import spaces
from gymnasium.vector.utils import batch_space

try:
    from ...render.navisim_scene import NavisimScene
except ImportError:
    import os, sys

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if root not in sys.path:
        sys.path.append(root)

    from navisim.render.navisim_scene import NavisimScene


@dataclass
class ControllerConfig:
    pass


class BaseController:
    """
    Base class for controllers.
    The controller is an interface for the agent to interact with the environment
    """

    action_space: spaces.Space

    def __init__(
        self,
        config: ControllerConfig,
        scene: NavisimScene,
        control_freq : int, 
        sim_freq: int,
    ):
        self.scene = scene
        self._control_freq = control_freq
        self._sim_steps = sim_freq // control_freq

    def _initialize_action_space(self):
        raise NotImplementedError
    
    @property
    def control_freq(self):
        return self._control_freq
    
    @property
    def pos(self):
        pass
    
    def set_action(self, action: List):
        pass