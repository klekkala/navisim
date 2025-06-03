import numpy as np

from typing import Tuple
from motion.base_motion_model import MotionModel

class SimpleMotionModel(MotionModel):
    def move(self, current_pose, action):
        x, y = current_pose[0], current_pose[1]
        dx, dy = action

        new_x = x + dx
        new_y = y + dy

        # Check boundaries
        if not self.sector.boundary.contains(new_x, new_y):
            return current_pose  # Invalid move

        # Update elevation from the elevation map
        z = self.sector.elevation_map.get_height_at(new_x, new_y)

        # Preserve orientation for now
        return (new_x, new_y, z, *current_pose[3:])