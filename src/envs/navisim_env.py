import gymnasium as gym
import numpy as np
import random
import pygame
import time
import torch

from gymnasium import spaces
from envs.game_window import GameWindow
from enum.enums import RelativeDir
from motion.motion_model import MotionModel
from spaces.sequence_graph import SequenceGraph

class NavisimEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, sequence_graph: SequenceGraph, render_mode=None, game_window=None):
        super().__init__()
        self.sequence_graph = sequence_graph
        self.render_mode = render_mode
        self.game_window = game_window

        self.current_sequence_id = None
        self.current_sector = None
        self.motion_model = None
        self.agent_pose = None
        self.goal_pose = None

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0, high=255, shape=(1090, 1959, 3), dtype=np.uint8),
            "aux": spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        })

        self.last_step = -1
        self.curr_step = 0
        self.offset_x = 0
        self.offset_y = 0
        self.grid_resolution = 1
        self.agent_path = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_sequence_id = random.choice(self.sequence_graph.get_sequence_ids())
        sequence = self.sequence_graph.get_sequence(self.current_sequence_id)
        self.current_sector = sequence[0]

        self._load_motion_model_for_current_sector()

        self.agent_pose = self._random_pose_in_sector(self.current_sector)
        self.goal_pose = self._random_goal_pose(self.current_sector)
        self.agent_path = [self.agent_pose]

        self.last_step = -1
        self.curr_step = 0

        return np.array(self.agent_pose, dtype=np.float32), {}

    def step(self, action):
        self.curr_step += 1

        new_pose = self.motion_model.move(self.agent_pose, action)

        if not self.current_sector.boundary.contains(new_pose[0], new_pose[1]):
            rel_dir = self.current_sector.boundary.direction_to(new_pose[0], new_pose[1])
            if rel_dir == RelativeDir.OUTSIDE_LEFT and self.current_sector.prev:
                self.current_sector = self.current_sector.prev
            elif rel_dir == RelativeDir.OUTSIDE_RIGHT and self.current_sector.next:
                self.current_sector = self.current_sector.next
            self._load_motion_model_for_current_sector()

        self.agent_pose = new_pose
        self.agent_path.append(self.agent_pose)

        obs = np.array(self.agent_pose, dtype=np.float32)
        reward, done = self._compute_reward_and_done()

        return obs, reward, done, False, {}

    def _load_motion_model_for_current_sector(self):
        self.motion_model = MotionModel(self.current_sector.elevation_map)

    def _random_pose_in_sector(self, sector):
        x, y = 1.0, 1.0
        z = sector.elevation_map.get_height_at(x, y)
        return (x, y, 0.0, z)

    def _random_goal_pose(self, sector):
        x, y = 8.0, 8.0
        z = sector.elevation_map.get_height_at(x, y)
        return (x, y, 0.0, z)

    def _compute_reward_and_done(self):
        dist = np.linalg.norm(np.array(self.agent_pose[:2]) - np.array(self.goal_pose[:2]))
        return -dist, dist < 0.5

    def _render_agent_camera(self):
        h, w = 100, 100
        return torch.randint(0, 255, (h, w, 3), dtype=torch.uint8)

    def render_loop(self):
        window = self.game_window or GameWindow()

        while window.running:
            if self.last_step != self.curr_step:
                start_time = time.time()
                self.last_step = self.curr_step

                tensorImg = self._render_agent_camera()
                elevation_map = self.current_node.elevation_map.map
                window.display_images(tensorImg, elevation_map)

                elapsed_time = time.time() - start_time
                fps = 1 / elapsed_time if elapsed_time > 0 else 0
                print(f"Total Seconds: {elapsed_time:.3f}")
                print(f"Frames per second: {fps:.2f}\n")

            self.process_keyboard_input(window)

        window.quit()

    def process_keyboard_input(self, window):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                window.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    print("Quit key (Q) pressed")
                    window.running = False
                else:
                    if event.key == pygame.K_w:
                        action = [1, 1]
                    elif event.key == pygame.K_a:
                        action = [1, -1]
                    elif event.key == pygame.K_s:
                        action = [-1, -1]
                    elif event.key == pygame.K_d:
                        action = [-1, 1]
                    else:
                        action = [0, 0]
                    self.step(action)

    def render(self):
        if self.render_mode == "human":
            print(f"Agent Pose: {self.agent_pose}, Goal: {self.goal_pose}")

    def close(self):
        pass