import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces

from motion.motion_model import MotionModel
from spaces.sequence_graph import SequenceGraph

class NavisimEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_sequence_id = random.choice(self.sequence_graph.get_sequence_ids())
        sequence = self.sequence_graph.get_sequence(self.current_sequence_id)
        self.current_sector = sequence[0]

        elevation_map = self.current_sector.elevation_map
        self.motion_model = MotionModel(elevation_map)

        self.agent_pose = self._random_pose_in_sector(self.current_sector)
        self.goal_pose = self._random_goal_pose(self.current_sector)

        return np.array(self.agent_pose, dtype=np.float32), {}

    def step(self, action):
        new_pose = self.motion_model.move(self.agent_pose, action)

        if not self.current_sector.boundary.contains(new_pose[0], new_pose[1]):
            new_pose = self.agent_pose

        self.agent_pose = new_pose
        obs = np.array(self.agent_pose, dtype=np.float32)

        reward, done = self._compute_reward_and_done()

        return obs, reward, done, False, {}

    def _random_pose_in_sector(self, sector):
        x, y = 1.0, 1.0  # Replace with smarter sampling
        z = sector.elevation_map.get_height_at(x, y)
        return (x, y, 0.0, z)

    def _random_goal_pose(self, sector):
        x, y = 8.0, 8.0
        z = sector.elevation_map.get_height_at(x, y)
        return (x, y, 0.0, z)

    def _compute_reward_and_done(self):
        dist = np.linalg.norm(np.array(self.agent_pose[:2]) - np.array(self.goal_pose[:2]))
        return -dist, dist < 0.5

    def render(self):
        if self.render_mode == "human":
            print(f"Agent Pose: {self.agent_pose}, Goal: {self.goal_pose}")

    def close(self):
        pass