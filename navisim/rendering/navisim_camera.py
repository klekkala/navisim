import os, sys

# Get the absolute path to your project root
# Assuming your structure is: project_root/experimental/test.ipynb
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))

# Add the third_party/gs directory to Python path
third_party_path = os.path.join(project_root, 'third_party', 'gaussian-splatting')
if third_party_path not in sys.path:
    sys.path.insert(0, third_party_path)
from scene.customCameras import CustomCamera

import numpy as np

class NavisimCamera(CustomCamera):
    def __init__(self, camera_id: str, position: list[float], rotation: list[float], W=1959, H=1090, FoVx = 1.4, FoVy = 0.87):
        """
        Initialize a NavisimCamera instance.

        Args:
            position (list[float]): Camera position in 3D space.
            rotation (list[float]): Camera rotation as Euler angles (pitch, yaw, roll).
            fov (float): Field of view in degrees.
        """
        self.camera_id = camera_id
        super().__init__(R = rotation, T=position, FoVx=FoVx, FoVy=FoVy, W=W, H=H)
    
    @classmethod
    def create(
        cls, 
        camera_id: str,
        yaw = 0, 
        pitch = 0, 
        roll = 0, 
        x = 0, 
        y= 0, 
        z = 0, 
        FoVx: float = 1.4,
        FoVy: float = 0.87,
        W: int = 1280,
        H: int = 720,
    ):
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)
        roll = np.radians(roll)

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

        position = np.array([x, y, z])
        viewpoint_camera = NavisimCamera(camera_id=camera_id, rotation=rotation, position=position, FoVx=FoVx, FoVy=FoVy, W=W, H=H)
        return viewpoint_camera


    def translate(self, x, y, z):
        self.trans[0] += x  # moving x-axis left and right
        self.trans[1] += y  # moving y-axis up or down
        self.trans[2] += z  # moving z-axis front and back
        self.update_transforms()
    
    def rotate(self, roll, pitch, yaw):
        R_x = self.rotation_matrix_x(pitch)
        R_y = self.rotation_matrix_y(yaw)
        R_z = self.rotation_matrix_z(roll)

        self.R = np.dot(self.R, R_x)
        self.R = np.dot(self.R, R_y)
        self.R = np.dot(self.R, R_z)

        self.update_transforms()
    
    def rotation_matrix_x(self, angle_radians):
        R_new = np.array([[1, 0, 0],
                        [0, np.cos(angle_radians), -np.sin(angle_radians)],
                        [0, np.sin(angle_radians), np.cos(angle_radians)]])
        return R_new

    def rotation_matrix_y(self, angle_radians):
        R_new = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                        [0, 1, 0],
                        [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
        return R_new

    def rotation_matrix_z(self, angle_radians):
        R_new = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                        [np.sin(angle_radians), np.cos(angle_radians), 0],
                        [0, 0, 1]])
        return R_new