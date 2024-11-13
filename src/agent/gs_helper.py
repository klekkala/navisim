import torch
import torchvision
from torchvision import transforms
import numpy as np
import os
from submodules.gaussian.gaussian_renderer import render
from submodules.gaussian.scene.customCameras import CustomCamera


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

def render_pose(view, gaussians, background, pose):
    translate(view, pose[0], pose[1], pose[2])
    rotate(view, pose[3], pose[4], pose[5])
    tensor = render(view, gaussians, CustomPipeline, background, scaling_modifier=1)["render"]

    # Make sure tensor is between 0 and 1
    tensor = torch.clamp(tensor, 0, 1)

    # Convert tensor to 0-255 and to byte format on GPU, then transfer to CPU
    tensor = (tensor * 255).byte().cpu()

    tensor = tensor.permute(1, 2, 0).contiguous()  # Change from CxHxW to HxWxC
    return tensor

def resize_tensor_img(tensor_img):
    # Resizing the images before sending to CPU
    new_height = 109
    new_width = 196
    resize_transform = transforms.Resize((new_height, new_width))
    image_resized = resize_transform(tensor_img.unsqueeze(0))  # Add batch dimension
    image_resized = image_resized.squeeze(0)  # Remove batch dimension
    return image_resized

def save_render_disk(tensor_img, render_path, idx):
    # render_path = os.path.join(self.render_output_path, self.render_output_folderName, "images")
    # makedirs(render_path, exist_ok=True)
    torchvision.utils.save_image(tensor_img, os.path.join(render_path, f"{idx}.png"))


def create_camera():
    yaw = np.radians(0)
    pitch = np.radians(0)
    roll = np.radians(0)

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