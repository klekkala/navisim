"""Custom gym spaces and enums for Navisim."""

from enum import Enum


class RenderMode(Enum):
    """Rendering modes for environments."""
    HUMAN = "human"
    RGB_ARRAY = "rgb_array"
    DEPTH_ARRAY = "depth_array"