import torch
import torchvision
import os
from os import makedirs
import numpy as np
from submodules.gaussian.scene import Scene
from submodules.gaussian.gaussian_renderer import render
from submodules.gaussian.gaussian_renderer import GaussianModel
from submodules.gaussian.scene.customCameras import CustomCamera

class Agent:
    def __init__(self, elevation_map, grid_resolution, elevation_x_offset, elevation_y_offset, min_height, start_location):
        self.start_location = start_location
        start_x, start_z = start_location

        # Gaussian Splatting Initialization
        self.model_path = './src/assets/gaussian_input'
        self.render_output_path = './src/assets/'
        self.render_output_folderName = 'gaussian_output'
        self.global_pose = [start_x, 0, start_z, 0, 0, np.pi]
        self.local_pose = [0, 0, 0, 0, 0, np.pi]
        self.curr_step = 0
        self.initScene()

        # Elevation Map Initialization
        self.elevation_map = elevation_map
        self.elevation_map_size = elevation_map.shape[0]  # The size of the elevation map
        self.elevation_map_height = elevation_map.shape[1]
        self.elevation_x_offset = elevation_x_offset
        self.elevation_y_offset = elevation_y_offset
        self.grid_resolution = grid_resolution
        self.min_height = min_height

        # Motion Model Initialization
        self.metadata = {"render_modes": ["human", "rgb_array"], "agent_size": 0.75, "agent_height": 1.75}
        self.current_vl = 0  # The initial velocity of the left wheel
        self.current_vr = 0  # The initial velocity of the left wheel
        self.agent_size = self.metadata["agent_size"]
        self.agent_height = self.metadata["agent_height"]
        self.agent_path = []
        self.isCollided = False

    def initScene(self):
        with torch.no_grad():
            self.gaussians = GaussianModel(3)
            modelParams = CustomModelParams(self.model_path)
            self.scene = Scene(modelParams, self.gaussians, load_iteration=-1, shuffle=False)
            bg_color = [0, 0, 0]
            self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    def reset(self, pos=None):
        self.curr_step = 0
        start_x, start_z = self.start_location
        self.global_pose = [start_x, 0, start_z, 0, 0, np.pi]
        self.local_pose = [0, 0, 0, 0, 0, np.pi]
        self.agent_path = []

    # Above contains all functions related to initializations of Agent
    # ----------------------------------------------------------------------------------------------------------

    # Help accessing elevation data from global elevation map.
    def returnElevationData(self, x, y):
        elevation_x = self.grid_resolution * np.abs(int(x) - self.elevation_x_offset)
        elevation_y = self.grid_resolution * np.abs(int(y) - self.elevation_y_offset)

        return self.elevation_map[elevation_x][elevation_y]
    
    # Generates images using gaussian splatting according to current local pose.
    def render_camera(self):
        render_path = os.path.join(self.render_output_path, self.render_output_folderName, "images")
        makedirs(render_path, exist_ok=True)

        camera = create_camera()
        render_pose(camera, self.gaussians, self.background, render_path, self.local_pose, self.curr_step)

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
        self.local_pose[1] = -self.global_pose[1] / 10
        # self.local_pose[1] = 2

        self.curr_step += 1

        # update path for visualizations
        self.agent_path.append([self.global_pose[0], self.global_pose[2], self.global_pose[1]])

        if (len(self.agent_path) >= 2 and ((self.agent_path[-1][2] - self.agent_path[-2][2]) > 0.2)):
            self.isCollided = True

        # Log
        print("Current Global Pose: ", self.global_pose)
        print("Current Local Pose: ", self.local_pose)
        print("time_elapsed: ", time_elapsed)
        return self.global_pose
    
    # not sure about this function is necessary
    def updateScene(self, elevation_map, offset_x, offset_y):
        #TODO update scene
        return
    # ----------------------------------------------------------------------------------------------------------
    
    def update_global_pose(self):
        # Temporary trans. Needs to dynamically reads the trans later if when switching sectors

        # Splat 1
        trans = np.array([
            [-7.28487262e-01,  3.00009073e-03,  6.85052778e-01],
            [-1.47005104e-02, -9.99828607e-01, -1.12539621e-02],
            [ 6.84901602e-01, -1.82689935e-02,  7.28406507e-01],
            [-7.27663079e+01, -4.59237964e+00,  1.74163156e+02]
        ])

        # Splat 2
        # trans = np.array([
        #     [ 9.67962430e-01,  8.61912865e-02,  2.35838497e-01],
        #     [ 6.23014016e-02, -9.92310474e-01,  1.06950734e-01],
        #     [ 2.43243232e-01, -8.88312230e-02, -9.65889095e-01],
        #     [-6.85761898e+01, -4.74879110e+00,  1.86610921e+02]
        # ])
        
        translations = np.dot(trans[:-1], np.array([self.local_pose[0], self.local_pose[1], self.local_pose[2]])) + trans[-1]

        self.global_pose[0] = translations[0] # x
        self.global_pose[2] = translations[2] # z
        self.global_pose[1] = self.returnElevationData(self.global_pose[0], self.global_pose[2]) # y

        self.global_pose[5] = self.local_pose[5]
        self.global_pose[3] = self.estimate_roll(self.global_pose[0], self.global_pose[2], self.agent_size, self.local_pose[5]) % (2 * np.pi)
        self.global_pose[4] = self.estimate_pitch(self.global_pose[0], self.global_pose[2], self.agent_size, self.local_pose[5]) % (2 * np.pi)


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


    def estimate_roll(self, x_prime, z_prime, l, yaw):
        # ϕ=arctan(left and right wheel height difference / distance between two point)
        l_wheel_x = x_prime + (-l / 2) * np.sin(yaw)
        l_wheel_y = z_prime + (l / 2) * np.cos(yaw)
        r_wheel_x = x_prime - (-l / 2) * np.sin(yaw)
        r_wheel_y = z_prime - (l / 2) * np.cos(yaw)

        # print(l_wheel_x, x_prime)
        # print(l_wheel_y, z_prime)
        # return 1
    
        l_elevation = self.returnElevationData(l_wheel_x, l_wheel_y)
        r_elevation = self.returnElevationData(r_wheel_x, r_wheel_y)

        roll = np.arctan((l_elevation - r_elevation) / l)

        return roll
    

    def estimate_pitch(self, x_prime, z_prime, l, yaw):
        # ϕ=arctan(front and back height difference / distance between two point)
        front_x = x_prime + (l / 2) * np.cos(yaw)
        front_z = z_prime + (l / 2) * np.sin(yaw)
        rear_x = x_prime - (l / 2) * np.cos(yaw)
        rear_z = z_prime - (l / 2) * np.sin(yaw)

        front_elevation = self.returnElevationData(front_x, front_z)
        rear_elevation = self.returnElevationData(rear_x, rear_z)

        pitch = np.arctan2((front_elevation - rear_elevation), l)
        return pitch

# Maybe move below code into another file to make the code clean
# Gaussian Splatting Helpfer Class & Functions
class CustomModelParams():
    def __init__(self, model_path):
        self.model_path = model_path

class CustomPipeline:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False

def rotation_matrix_x(angle_radians):
    R_new = np.array([[1, 0, 0],
                      [0, np.cos(angle_radians), -np.sin(angle_radians)],
                      [0, np.sin(angle_radians), np.cos(angle_radians)]])
    return R_new

def rotation_matrix_y(angle_radians):
    R_new = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                      [0, 1, 0],
                      [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
    return R_new

def rotation_matrix_z(angle_radians):
    R_new = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                      [np.sin(angle_radians), np.cos(angle_radians), 0],
                      [0, 0, 1]])
    return R_new

def rotate(view, roll, pitch, yaw):
    R_x = rotation_matrix_x(pitch)
    R_y = rotation_matrix_y(yaw)
    R_z = rotation_matrix_z(roll)

    view.R = np.dot(view.R, R_x)
    view.R = np.dot(view.R, R_y)
    view.R = np.dot(view.R, R_z)

    view.update_transforms()

def translate(view, x, y, z):
    view.trans[0] += x  # moving x-axis left and right
    view.trans[1] += y  # moving y-axis up or down
    view.trans[2] += z  # moving z-axis front and back
    view.update_transforms()

def render_pose(view, gaussians, background, render_path, pose, idx=0):
    translate(view, pose[0], pose[1], pose[2])
    rotate(view, pose[3], pose[4], pose[5])
    rendering = render(view, gaussians, CustomPipeline, background)["render"]
    torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx}.png"))

def create_camera():
    # rotation = np.array([[0.8761342012188561, 0.06925962026449778, 0.47706599800804744],
    #                      [-0.04747421839895103, 0.9972110940209488, -0.05758673934988209],
    #                      [-0.4797239414934442, 0.02780537650095985, 0.8769787916452907]])
    
    # rotation = np.array([[0.9802084581685495, 0.00012913566697058749, 0.19796808292958304],
    #                      [-0.00010434069063855217, 0.9999999853521653, -0.00013567862650630828],
    #                      [-0.1979680975507292, 0.00011233721079691208, 0.9802084572847227]])

    pitch = np.radians(0)
    yaw = np.radians(0)
    roll = np.radians(5)
    
    # Rotation matrix for rotation around x-axis
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch), np.cos(pitch)]])
    
    # Rotation matrix for rotation around y-axis
    R_y = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                    [0, 1, 0],
                    [-np.sin(yaw), 0, np.cos(yaw)]])
    
    # Rotation matrix for rotation around z-axis
    R_z = np.array([[np.cos(roll), -np.sin(roll), 0],
                    [np.sin(roll), np.cos(roll), 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix
    rotation = R_z @ R_y @ R_x

    position = np.array([0, 0, 0])
    view = CustomCamera(R=rotation, T=position, FoVx=1.4, FoVy=0.87, W=1959, H=1090)
    return view
