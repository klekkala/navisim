import uuid
import gymnasium as gym
import numpy as np
import random

from navisim.config.gaussian_model_param import GaussianModelParam

from navisim.enums.enums import RenderMode
from navisim.envs.game_window import GameWindow
from navisim.motion.simple_motion_model import SimpleMotionModel
from navisim.rendering.navisim_camera import NavisimCamera
from navisim.rendering.navisim_scene import NavisimScene
from navisim.world.sequence_graph import SequenceGraph

import time

class NavisimEnv(gym.Env):
    def __init__(self, sequence_graph : SequenceGraph, render_mode: RenderMode, window : GameWindow):
        self.render_mode = render_mode.name.lower()
        self.sequence_graph = sequence_graph
        
        self.agent_pose = None
        self.goal_pose = None
        self.current_step = 0
        self.agent_path = []

        # TODO: Define action space
        self.action_space = gym.spaces.Discrete(3)

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = window
        self.sequence_index = 0
        self.current_sector = None
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_sequence_id = random.choice(self.sequence_graph.get_sequence_ids())
        sequence = self.sequence_graph.get_sequence(self.current_sequence_id)
        if self.current_sector:
            self.current_sector.unload_all()

        self.current_sector = sequence[self.sequence_index]

        self.motion_model = self._load_motion_model_for_current_sector(self.current_sector)
        self.scene = self._load_scene_for_current_sector(self.current_sector)
        
        self._agent_pose = self.scene.random_start_location()

        # set target location as the occupancy map
        self._target_locations = self.scene.get_target_locations()
        self.current_step = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == RenderMode.HUMAN.name.lower():
            self._render_frame()

        return observation, info

    #TODO: Update motion model for the env
    def _load_motion_model_for_current_sector(self, sector):
        return SimpleMotionModel()
    
    def _load_scene_for_current_sector(self, sector):
        return NavisimScene.create(
            model_params=GaussianModelParam.create(
                model_path=sector.gaussian_model.model_path
            ),
            camera = NavisimCamera.create(
                camera_id=str(uuid.uuid4()),
                    W = self.window.width,
                    H = self.window.height
                ),
            sector=sector
        )

    def step(self, action):
        if self.current_step % 10 == 0:
            self.sequence_index += 1
            self.reset()

        self.current_step += 1
        self._agent_pose = self.motion_model.move(self._agent_pose, action)

        terminated = False
        reward = 1 if terminated else 0
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == RenderMode.HUMAN.name.lower():
            _, elapsed = self._render_frame()
            info['elapsed'] = elapsed
            
        return obs, reward, terminated, False, info

    def step_with_motion_model(self, action):
        self.current_step += 1
        new_pose = self.motion_model.move(self.agent_pose, action)

        # Update render output
        self.update_pose(action)

        # TODO: sector updates with respect to Agent's pose
        raise NotImplementedError("Sector change is not implemented yet")
        # if not self.current_sector.boundary.contains(new_pose[0], new_pose[1]):
        #     rel_dir = self.current_sector.boundary.direction_to(new_pose[0], new_pose[1])
        #     if rel_dir == "OUTSIDE_LEFT" and self.current_sector.prev:
        #         self.current_sector = self.current_sector.prev
        #     elif rel_dir == "OUTSIDE_RIGHT" and self.current_sector.next:
        #         self.current_sector = self.current_sector.next
        #     self._load_motion_model_for_current_sector()

        terminated = False
        reward = 1 if terminated else 0
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == RenderMode.HUMAN.name.lower():
            self._render_frame()
        return obs, reward, terminated, False, info

    
    def _get_obs(self):
        return {"agent": self._agent_pose, "target": self._target_locations}

    def _get_info(self):
        agent_xy = self._agent_pose[:2]  # just x and y
        targets = np.array(self._target_locations)  # shape (213, 2)

        distances = np.linalg.norm(targets - agent_xy, ord=1, axis=1)
        closest = np.min(distances)

        return {
            "distance": closest,
            "num_targets": len(targets),
            "agent": tuple(self._agent_pose)
        }


    def render(self):
        return self._render_frame()
        
    def _render_frame(self):
        start = time.time()
        rendering = self.scene.render_from_camera(self._agent_pose)
        elapsed = time.time() - start

        self.window.display_images(rendering, None)

        return self.window.get_window_pixels(), elapsed
    
    def close(self):
        pass
