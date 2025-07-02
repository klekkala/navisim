import numpy as np

from enum import Enum
from scipy.spatial.transform import Rotation as R

try:
    from .base_motion_model import MotionModel
except ImportError:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from navisim.motion.base_motion_model import MotionModel

class Actions(Enum):
    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    
class SimpleMotionModel(MotionModel):
    def __init__(self):
        self._action_to_direction = {
            Actions.FORWARD.value: np.array([1.0, 0.0, 0.0]),     # move forward, no rotation
            Actions.LEFT.value:    np.array([0.0, 0.0, np.pi/12]), # rotate left (CCW)
            Actions.RIGHT.value:   np.array([0.0, 0.0, -np.pi/12]) # rotate right (CW)
        }

    def move(self, agent_pose, action):
        x, y, z, yaw, pitch, roll = agent_pose

        # Get intended movement (e.g., [forward, lateral, delta_yaw])
        direction = self._action_to_direction[action]
        dx_local, dy_local, dyaw = direction

        # Update yaw (rotation around Z axis)
        yaw = (yaw + dyaw) % (2 * np.pi)

        # Transform local direction to global coordinates
        r = R.from_euler('z', yaw)
        delta_global = r.apply([dx_local, dy_local, 0.0])  # No change in z

        # Update position (clip to boundary)
        new_position = np.array([x, y, z]) + delta_global
        new_position = np.clip(new_position, [0, 0, z], [1000, 1000, z])  # clamp x and y only

        # Update pose
        return [*new_position, yaw, pitch, roll]