import gymnasium as gym
from gymnasium import spaces
import cv2
import os
import numpy as np
from src.beogym.pointcloud.pointcloud import *
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
        self.current_node = next(iter(self.sequence_graph))
        current_point_cloud = self.current_node.point_cloud
        self.elevation_map = current_point_cloud.elevation_map
        self.grid_resolution = current_point_cloud.grid_resolution
        self.offset_x, self.offset_y, self.min_height = current_point_cloud.get_elevation_map_info()
        
        self.elevation_map_size = self.elevation_map.shape[0]  # The size of the elevation map
        self.elevation_map_height = self.elevation_map.shape[1]

        assert 0 <= start_location[0] <= self.elevation_map_size or 0 <= start_location[1] <= self.elevation_map_size, f'Startting cooridnate should be within the boundaries of the elevation map({self.elevation_map_size}x{self.elevation_map_size}:{start_location})'
        assert 0 <= start_location[0] <= self.elevation_map_size or 0 <= start_location[1] <= self.elevation_map_size, f'Startting cooridnate should be within the boundaries of the elevation map({self.self.elevation_map_size}x{self.elevation_map_size})'
        assert 0 <= target_location[0] <= self.elevation_map_size or 0 <= target_location[1] <= self.elevation_map_size, f'Target cooridnate should be within the boundaries of the elevation map({self.elevation_map_size}x{self.elevation_map_size})'

        self.agent = Agent(self.elevation_map,
                           grid_resolution=self.grid_resolution,
                           elevation_x_offset=self.offset_x,
                           elevation_y_offset=self.offset_y,
                           min_height=self.min_height,
                           start_location=start_location)
        
        # visualize_3d(point_cloud)


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
        return self.agent.isCollided
        agent_current_coo = pose[[0, 2]].astype(int)
        has_reached_target_loc = np.array_equal(agent_current_coo, self._target_location)

        collided = self.is_collided(pose)

        if has_reached_target_loc:
            print('Reached target location')
        
        if collided:
            print('Agent collided')
        
        return has_reached_target_loc or collided


    def step(self, action):
        estimated_pose = self.agent.take_action(action)

        x, y, z = estimated_pose[:3]
        self.check_graph((x,y,z))

        # An episode is done iff the agent has reached the target
        terminated = self.is_terminated(estimated_pose)
        truncated = False
        reward = 0 if terminated else 0  # TODO(jiwon-hae): implement reward method later
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


    def process_keyboard_input(self):
        terminate = False

        class KeyMap:
            left = ord('a')
            right = ord('d')
            forward = ord('w')
            reverse = ord('s')
            terminate = ord('k')

        print("Enter operation (Left, Right, Forward, Reverse, K to kill): ")
        key = cv2.waitKeyEx(0)
        
        action = [0, 0]
        if key == KeyMap.left:
            print("Turn Left")
            action = [1, -1]
        elif key == KeyMap.right:
            print("Turn Right")
            action = [-1, 1]
        elif key == KeyMap.forward:
            print("Moving Forward")
            action = [1, 1]
        elif key == KeyMap.reverse:
            print("Moving Reverse")
            action = [-1, -1]
        elif key == KeyMap.terminate:
            print("Terminating...")
            terminate = True

        self.step(action)
        return terminate


    def display_image(self, image_path):
        try:
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    window_title = f'Image Window - Step {self.agent.curr_step}'
                    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)  # Create a resizable window
                    cv2.resizeWindow(window_title, 640, 480)  # Resize the window to 640x480 pixels
                    cv2.moveWindow(window_title, 1920 - 640, (1080 - 480) // 2)
                    cv2.imshow(window_title, image)
                    cv2.waitKey(1)  # Refresh to display the image
                else:
                    print(f"Failed to load image at {image_path}")
            else:
                print(f"No file found at {image_path}")
        except Exception as e:
            print(f"An error occurred: {e}")


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


    def render(self):
        if self.render_mode == "human":
            while True:
                terminate_program = self.process_keyboard_input()
                cv2.destroyAllWindows()
                plt.close('all')
                if terminate_program:
                    break
                else:
                    self.agent.render_camera()
                    image_path = './src/assets/gaussian_output/images/' + str(self.agent.curr_step) + '.png'
                    self.display_image(image_path)

                    plot_elevation_map(elevation_map=self.elevation_map,
                                       withCV2=True,
                                       shift_x= self.offset_x,
                                       shift_y = self.offset_y,
                                       grid_resolution=self.grid_resolution,
                                       agent_location=self.agent.agent_path,
                                       save_path=f'./src/output/elevation')

                    # plot_occupnacy_map(occupancy_map=self.elevation_map,
                    #                    withCV2=True,
                    #                    shift_x= self.offset_x,
                    #                    shift_y = self.offset_y,
                    #                    grid_resolution=self.grid_resolution,
                    #                    agent_location=self.agent.agent_path,
                    #                    save_path=f'./src/output/occupancy')
        else:
            self.agent.render_camera()


