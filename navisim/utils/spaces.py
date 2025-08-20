"""Custom gym spaces and enums for Navisim."""

from enum import Enum, auto


class RenderMode(Enum):
    """Rendering modes for environments."""
    NONE = auto()
    HUMAN = "human"
    RGB_ARRAY = "rgb_array"
    DEPTH_ARRAY = "depth_array"


class RlPolicy(Enum):
    """Reinforcement learning policies."""
    RANDOM = auto()


class RelativeDir(Enum):
    """Relative direction enums."""
    INSIDE_OR_ON = auto()
    OUTSIDE_TOP = auto()
    OUTSIDE_RIGHT = auto()
    OUTSIDE_BOTTOM = auto()
    OUTSIDE_LEFT = auto()