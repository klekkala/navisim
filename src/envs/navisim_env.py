import gymnasium as gym
import random

from motion.base_motion_model import MotionModel
from rendering.navisim_scene import NavisimScene
from config.gaussian_model_param import GaussianModelParam
from world.sequence_graph import SequenceGraph

class NavisimEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, sequence_graph : SequenceGraph, render_mode=None):
        self.render_mode = render_mode
        self.sequence_graph = sequence_graph
        
        self.agent_pose = None
        self.goal_pose = None
        self.step_count = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_sequence_id = random.choice(self.sequence_graph.get_sequence_ids())
        sequence = self.sequence_graph.get_sequence(self.current_sequence_id)
        self.current_sector = sequence[0]

        self.motion_model = self._load_motion_model_for_current_sector(self.current_sector)
        self.scene = self._load_scene_for_current_sector(self.current_sector)

        self.agent_pose = self.scene.random_start_pose()
        self.goal_pose = self.scene.random_goal_pose()
        self.step_count = 0

        return self.scene.build_observation(self.scene, self.agent_pose, self.goal_pose), {}

    #TODO: Update motion model for the env
    def _load_motion_model_for_current_sector(self, sector):
        return MotionModel(sector)
    
    def _load_scene_for_current_sector(self, sector):
        return NavisimScene.create(
            model_params=GaussianModelParam.create(
                path=sector.gaussian_model.model_path
            ),
            sector=sector
        )

    def step(self, action):
        self.curr_step += 1
        new_pose = self.motion_model.move(self.agent_pose, action)

        if not self.current_sector.boundary.contains(new_pose[0], new_pose[1]):
            rel_dir = self.current_sector.boundary.direction_to(new_pose[0], new_pose[1])
            if rel_dir == "OUTSIDE_LEFT" and self.current_sector.prev:
                self.current_sector = self.current_sector.prev
            elif rel_dir == "OUTSIDE_RIGHT" and self.current_sector.next:
                self.current_sector = self.current_sector.next
            self._load_motion_model_for_current_sector()

        self.agent_pose = new_pose
        self.agent_path.append(new_pose)

        obs = self.scene.build_observation(self.agent_pose, self.goal_pose)
        reward, done = self._compute_reward_and_done()
        return obs, reward, done, False, {}

    def _compute_reward_and_done(self):
        raise NotImplementedError("Reward and done logic should be implemented in the subclass")

    def render(self):
        raise NotImplementedError("Rendering is not implemented yet")
        # if self.render_mode == "human":
        #     img = self.scene.render_from_camera_pose(self.agent_pose)
            # display img with OpenCV or matplotlib

    def close(self):
        pass
