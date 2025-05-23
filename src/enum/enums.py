from enum import Enum, auto

class RelativeDir(Enum):
    INSIDE_OR_ON = auto()
    OUTSIDE_TOP = auto()
    OUTSIDE_RIGHT = auto()
    OUTSIDE_BOTTOM = auto()
    OUTSIDE_LEFT = auto()