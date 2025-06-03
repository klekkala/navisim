from gaussian_splatting.scene.customCameras import CustomCamera

import numpy as np

class NavisimCamera(CustomCamera):
    def __init__(self, camera_id: str, position: list[float], rotation: list[float], fov: float = 60.0):
        """
        Initialize a NavisimCamera instance.

        Args:
            camera_id (str): Unique identifier for the camera.
            position (list[float]): Camera position in 3D space.
            rotation (list[float]): Camera rotation as Euler angles (pitch, yaw, roll).
            fov (float): Field of view in degrees.
        """
        super().__init__(camera_id, position, rotation, fov)
    
    @classmethod
    def create(
        camera_id: str = "navisim_cam",
        yaw = 0, 
        pitch = 0, 
        roll = 0, 
        x = 0, 
        y= 0, 
        z = 0, 
        FoVx: float = 1.4,
        FoVy: float = 0.87,
        W: int = 1959,
        H: int = 1090,
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
        viewpoint_camera = CustomCamera(R=rotation, T=position, FoVx=FoVx, FoVy=FoVy, W=W, H=H)
        return viewpoint_camera


    def translate(self, x, y, z):
        self.trans[0] += x  # moving x-axis left and right
        self.trans[1] += y  # moving y-axis up or down
        self.trans[2] += z  # moving z-axis front and back
        self.update_transforms()
    
    def rotate(self, view, roll, pitch, yaw):
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