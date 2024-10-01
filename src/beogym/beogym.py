import gymnasium as gym
from gymnasium import spaces
import cv2
import os
import numpy as np
from src.beogym.game_window import *
from src.beogym.pointcloud.pointcloud import *
from src.beogym.visualization import *

# from pointcloud.elevation_map.elevation_map_3d_visualization import visualize_3d
import matplotlib.pyplot as plt
from src.agent.agent import Agent

class BeoGym(gym.Env):
    def __init__(self, config, sequence_graph, target_location, start_location = np.zeros(3, dtype=int), render_mode=None):
        self._start_location = start_location
        self._target_location = target_location
        self.render_mode = render_mode
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0, high=255, shape=(1090, 1959, 3), dtype=np.uint8),
            "aux": spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        })

        config = config or {}
        self.sequence_graph = sequence_graph
        self.current_node = self.get_node_from_sequence_graph(agent_location = start_location)
        # current_point_cloud = self.current_node.point_cloud

        #TODO (jiwon) : currently using global elevation map, will have to change to elevation map of each sector
        #self.elevation_map = current_point_cloud.elevation_map
        # self.elevation_map = self.sequence_graph.global_elevation_map
        # self.grid_resolution = current_point_cloud.grid_resolution
        # self.offset_x, self.offset_y, self.min_height = current_point_cloud.get_elevation_map_info()

        # assert 0 <= start_location[0] <= self.elevation_map_size or 0 <= start_location[1] <= self.elevation_map_size, f'Startting cooridnate should be within the boundaries of the elevation map({self.elevation_map_size}x{self.elevation_map_size}:{start_location})'
        # assert 0 <= start_location[0] <= self.elevation_map_size or 0 <= start_location[1] <= self.elevation_map_size, f'Startting cooridnate should be within the boundaries of the elevation map({self.self.elevation_map_size}x{self.elevation_map_size})'
        # assert 0 <= target_location[0] <= self.elevation_map_size or 0 <= target_location[1] <= self.elevation_map_size, f'Target cooridnate should be within the boundaries of the elevation map({self.elevation_map_size}x{self.elevation_map_size})'

        self.agent = Agent(node = self.current_node,
                           start_location=start_location)
    
    def get_node_from_sequence_graph(self, agent_location):
        """
        TODO(jiwon) : get node given the agent coordinate
        """
        agent_x, agent_z = agent_location
        return self.sequence_graph.get_node(agent_x, agent_z)

        
    def translate_elevation_index(self, x, y):
        return (self.grid_resolution * (x + self.offset_x), self.grid_resolution * (y + self.offset_y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent.reset()
        return self._get_obs()

    def _get_obs(self):
        return {"agent": self.agent.global_pose, "target": self._target_location}

    def _get_info(self):
        agent_location_x_z = np.array([self.agent.global_pose[0], self.agent.global_pose[2]])
        return {
            "distance": np.linalg.norm(agent_location_x_z - self._target_location, ord=1)
        }

    def is_terminated(self, pose):
        print("Collision status: ", self.agent.isCollided)
        
        def is_within_tolerance(value1, value2, tolerance = 0.2):
            return abs(value1 - value2) <= tolerance

        if is_within_tolerance(self._target_location[0], pose[0]) and is_within_tolerance(self._target_location[1], pose[2]):
            print('Reached target location')
            return True
        elif self.agent.isCollided:
            print('Agent collided')
            return True
        else:
            return False

    def step(self, action):
        estimated_pose = self.agent.take_action(action)

        # # Make below more efficient.
        # x, y, z = estimated_pose[:3]
        # self.check_graph((x,y,z))

        # An episode is done iff the agent has reached the target
        terminated = self.is_terminated(estimated_pose)
        truncated = False
        reward = 0 if terminated else 0  # TODO(jiwon-hae): implement reward method later
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def check_graph(self, agent_coordinates):
        x, y, z = agent_coordinates
        
        bound_x1, bound_y1, bound_x2, bound_y2 = self.current_node.boundary
        if bound_x1 >= x or bound_y1 >= z or bound_x2 <= x or bound_y2 <= z:
            self.update_agent_and_env(list(self.sequence_graph.neighbors(self.current_node))[0])

    # TODO(jiwon) : update scene and the splat file
    def update_agent_and_env(self, node):
        print('scene updated')
        self.current_node = node
        current_point_cloud = node.get_point_cloud()
        elevation_map = current_point_cloud.get_elevation_map()
        offset_x, offset_y, min_height = current_point_cloud.get_elevation_map_info()
        self.agent.updateScene(elevation_map, offset_x, offset_y)

    def process_keyboard_input(self, window):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                window.running = False
            
            # Check for key presses
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    print("Quit key (Q) pressed")
                    window.running = False  # Quit the loop
                else:
                    if event.key == pygame.K_w:
                        # print("Moving Forward")
                        action = [1, 1]
                    elif event.key == pygame.K_a:
                        # print("Moving Left")
                        action = [1, -1]
                    elif event.key == pygame.K_s:
                        # print("Moving Reverse")
                        action = [-1, -1]
                    elif event.key == pygame.K_d:
                        # print("Moving Right")
                        action = [-1, 1]
                    else:
                        action = [0, 0]
                    self.step(action)

    def render(self):
        import time
        if self.render_mode == "human":
            if (True):
                window = GameWindow()
                while window.running:
                    if (self.last_step != self.agent.curr_step):
                        print()
                        start_time = time.time()
                        self.last_step = self.agent.curr_step

                        tensorImg = self.agent.render_camera()
                        render_time = time.time() - start_time
                        print(f"Rendering time: {render_time}")
                        
                        elevImg = plot_elevation_map_io(elevation_map=self.elevation_map,
                                        saveOnly=True,
                                        shift_x= self.offset_x,
                                        shift_y = self.offset_y,
                                        grid_resolution=self.grid_resolution,
                                        agent_location=self.agent.agent_path,
                                        save_path=f'./src/output/elevation')
                        
                        # elevImg = None
                        window.display_images(tensorImg, elevImg)

                        # Calculate the elapsed time
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        fps = 1 / elapsed_time if elapsed_time > 0 else 0  # Added a check to avoid division by zero
                        print()
                        print(f"Total Seconds: {elapsed_time}")
                        print(f"Frames per second: {fps}")
                        print()

                    self.process_keyboard_input(window)
                window.quit()
            else:
                pass
        else:
            self.agent.render_camera()


