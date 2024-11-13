import torch
import os
import numpy as np
from submodules.gaussian.scene import Scene
from submodules.gaussian.gaussian_renderer import GaussianModel
from src.paths import *
from src.agent.gs_helper import *

class Agent:
    def __init__(self, node, start_location, target_location):
        self.start_location = start_location
        self.target_location = target_location
        start_x, start_z = start_location

        # Gaussian Splatting Initialization
        self.gs_splat_path = os.path.join(GAUSSIAN_SPLAT_FOLDER, 'gaussian_input')
        self.global_pose = [start_x, 0, start_z, 0, 0, np.pi]
        self.local_pose = [0, 0, 0, 0, 0, np.pi]
        self.curr_step = 0
        self.last_step = None
        self.initScene()

        # # Elevation Map Initialization
        self.elevation_map = node.elevation_map
        self.elevation_map_size = node.elevation_map.shape[0]  # The size of the elevation map
        self.elevation_map_height = node.elevation_map.shape[1]
        # self.elevation_x_offset = node.offset_x
        # self.elevation_y_offset = node.offset_y
        self.grid_resolution = node.grid_resolution
        # self.min_height = node.min_height
        self.trans_matrix = node.transformation_matrix

        # Motion Model Initialization
        self.metadata = {"render_modes": ["human", "rgb_array"], "agent_size": 0.75, "agent_height": 1.75}
        self.current_vl = 0  # The initial velocity of the left wheel
        self.current_vr = 0  # The initial velocity of the right wheel
        self.agent_size = self.metadata["agent_size"]
        self.agent_height = self.metadata["agent_height"]
        self.agent_path = []
        self.isCollided = False

    def initScene(self):
        with torch.no_grad():
            self.gaussians = GaussianModel(3)
            modelParams = CustomModelParams(self.gs_splat_path)
            self.scene = Scene(modelParams, self.gaussians, load_iteration=-1, shuffle=False)
            bg_color = [0, 0, 0]
            self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    def reset(self, pos=None):
        self.curr_step = 0
        start_x, start_z = self.start_location
        self.global_pose = [start_x, 0, start_z, 0, 0, 0]
        self.local_pose = [0, 0, 0, 0, 0, 0]
        self.agent_path = []

    # Help accessing elevation data from global elevation map.
    # def returnElevationData(self, x, y):
    #     elevation_x = self.grid_resolution * np.abs(int(x) - self.elevation_x_offset)
    #     elevation_y = self.grid_resolution * np.abs(int(y) - self.elevation_y_offset)
    #     return self.elevation_map[elevation_x][elevation_y]

    # Generates images using gaussian splatting according to current local pose.
    def render_camera(self):
        if self.last_step != self.curr_step:
            self.last_step = self.curr_step
            camera = create_camera()
            return render_pose(camera, self.gaussians, self.background, self.local_pose)

    # Take action function for RL environment.
    def take_action(self, action):
        # time_elapsed = 0.075 #seconds
        time_elapsed = 0.15 #seconds

        self.current_vl = action[0]
        self.current_vr = action[1]
        self.current_vl = max(-1, min(1, self.current_vl))
        self.current_vr = max(-1, min(1, self.current_vr))

        if (self.current_vl == self.current_vr):
            time_elapsed = 1 #seconds

        self.local_pose = self.estimate_pose(time_elapsed, self.local_pose)
        self.update_global_pose()

        # Update local pose again for the y (elevation), roll, and pitch, and yaw.
        # The below code needs to be changed
        self.local_pose[1] = -self.global_pose[1] / 10
        # self.local_pose[1] = 2

        self.curr_step += 1

        # update path for visualizations
        # x, z, y, yaw
        self.agent_path.append([self.global_pose[0], self.global_pose[2], self.global_pose[1], self.global_pose[5]])

        if (len(self.agent_path) >= 2 and ((self.agent_path[-1][2] - self.agent_path[-2][2]) > 0.2)):
            self.isCollided = True

        # Log
        print("Current Global Pose: ", self.global_pose)
        print("Current Local Pose: ", self.local_pose)
        print("time_elapsed: ", time_elapsed)
        return self.global_pose

    def update_global_pose(self):
        R = self.trans_matrix[:3]
        t = self.trans_matrix[-1]

        translations = np.dot(R, np.array([self.local_pose[0], self.local_pose[1], self.local_pose[2]])) + t
        yaw = np.arctan2(R[0, 2], R[2, 2])
        roll = np.arctan2(R[2, 1], R[0, 2])
        # # roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 0])

        self.global_pose[0] = translations[0] # x
        self.global_pose[2] = translations[2] # z
        self.global_pose[1] = 0
        # self.global_pose[1] = self.returnElevationData(self.global_pose[0], self.global_pose[2]) # y

        self.global_pose[5] = yaw - self.local_pose[5]

        # The below code is acutally not even applied..
        # self.global_pose[3] = self.estimate_roll(self.global_pose[0], self.global_pose[2], self.agent_size, self.local_pose[5]) % (2 * np.pi)
        # self.global_pose[4] = self.estimate_pitch(self.global_pose[0], self.global_pose[2], self.agent_size, self.local_pose[5]) % (2 * np.pi)


    def estimate_pose(self, time_elapsed, pose):
        x, y, z, roll, pitch, yaw = pose
        vl, vr = self.current_vl, self.current_vr

        if vl == vr:
            z_prime = z + vl * time_elapsed * np.cos(yaw)
            x_prime = x + vl * time_elapsed * np.sin(yaw)
            yaw_prime = yaw
        else:
            # rate of rotation
            omega = ((vr - vl) / self.agent_size)

            # signed distance from the ICC to the midpoint between the wheels
            R = (self.agent_size / 2.0) * ((vl + vr) / (vr - vl))

            # Calculate ICC position relative to the current pose
            ICC_z = z - R * np.sin(yaw)
            ICC_x = x + R * np.cos(yaw)

            # Rotation matrix
            rotation_matrix = np.array([
                [np.cos(omega * time_elapsed), -np.sin(omega * time_elapsed), 0],
                [np.sin(omega * time_elapsed), np.cos(omega * time_elapsed), 0],
                [0, 0, 1]
            ])

            # Pose vector
            pose_vector = np.array([z - ICC_z, x - ICC_x, yaw])

            # ICC vector
            ICC_vector = np.array([ICC_z, ICC_x, omega * time_elapsed])

            # Calculate new pose
            new_pose = np.dot(rotation_matrix, pose_vector) + ICC_vector
            
            z_prime, x_prime, yaw_prime = new_pose

        # roll_prime = self.estimate_roll(x_prime, z_prime, self.agent_size, yaw_prime)
        # pitch_prime = self.estimate_pitch(x_prime, z_prime, self.agent_size, yaw_prime)

        return np.array([x_prime, y, z_prime, roll % (2 * np.pi), pitch % (2 * np.pi), yaw_prime % (2 * np.pi)])


    # def estimate_roll(self, x_prime, z_prime, l, yaw):
    #     # ϕ=arctan(left and right wheel height difference / distance between two point)
    #     l_wheel_x = x_prime + (-l / 2) * np.sin(yaw)
    #     l_wheel_y = z_prime + (l / 2) * np.cos(yaw)
    #     r_wheel_x = x_prime - (-l / 2) * np.sin(yaw)
    #     r_wheel_y = z_prime - (l / 2) * np.cos(yaw)

    #     l_elevation = self.returnElevationData(l_wheel_x, l_wheel_y)
    #     r_elevation = self.returnElevationData(r_wheel_x, r_wheel_y)

    #     roll = np.arctan((l_elevation - r_elevation) / l)
    #     return roll
    

    # def estimate_pitch(self, x_prime, z_prime, l, yaw):
    #     # ϕ=arctan(front and back height difference / distance between two point)
    #     front_x = x_prime + (l / 2) * np.cos(yaw)
    #     front_z = z_prime + (l / 2) * np.sin(yaw)
    #     rear_x = x_prime - (l / 2) * np.cos(yaw)
    #     rear_z = z_prime - (l / 2) * np.sin(yaw)

    #     front_elevation = self.returnElevationData(front_x, front_z)
    #     rear_elevation = self.returnElevationData(rear_x, rear_z)

    #     pitch = np.arctan2((front_elevation - rear_elevation), l)
    #     return pitch


    # -----------------------------------------------------------------------------------------------------
    # Functions that might be Helpful
    # def translate_elevation_index(self, x, y):
    #     return (self.grid_resolution * (x + self.offset_x), self.grid_resolution * (y + self.offset_y))