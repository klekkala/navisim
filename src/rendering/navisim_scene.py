import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', '..')))

from config.gaussian_model_param import GaussianModelParam
from rendering.navisim_camera import NavisimCamera
from world.sector import Sector
from gaussian_splatting.scene import Scene
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.gaussian_renderer import GaussianModel

import numpy as np
import torch

class NavisimScene(Scene):
    def __init__(self, model_params, camera : NavisimCamera, gaussian_model : GaussianModel, sector : Sector, background, device="cuda"):
        super().__init__(
            model_params=model_params,
            gaussians=gaussian_model,
            load_iteration=-1,
            shuffle=False
        )

        self.sector = sector
        self.camera = camera
        self.gaussian_model = gaussian_model
        self.background = background
        self.device = device

    @classmethod
    def create(cls, model_params: GaussianModelParam, sector : Sector, camera: NavisimCamera = NavisimCamera.create()) -> "NavisimScene":
        """
        Factory method to create a NavisimScene instance using GaussianModelParam.

        Args:
            model_params (GaussianModelParam): Model config with path, SH degree, device, etc.

        Returns:
            NavisimScene: Initialized scene with loaded Gaussian model.
        """
        device = model_params.device
        gaussian_model = GaussianModel(sh_degree=model_params.sh_degree).to(device)
        resolved_params = model_params.extract(None)  # Or use actual parser args if needed
        background = torch.tensor(model_params.bg_color, dtype=model_params.dtype, device=device)
        return cls(resolved_params, camera, gaussian_model, sector, background, device)

    #TODO: needs to discuss how to determine the start pose
    def random_start_pose(self):
        """
        Generate a random starting pose within the scene.

        Returns:
            list: A list containing x, y, z, yaw, roll, pitch.
        """

        x, y = self.sector.boundary.sample_point_within()
        z = self.sector.elevation_map.get_height_at(x, y)
    
        yaw = torch.randint(0, 360, (1,)).item()
        roll = torch.randint(0, 360, (1,)).item()
        pitch = torch.randint(0, 360, (1,)).item()
        return [x, y, z, yaw, roll, pitch]
    
    def random_goal_pose(self):
        """
        Generate a random goal pose within the scene.

        Returns:
            list: A list containing x, y, z, yaw, roll, pitch.
        """
        x, y = self.sector.boundary.sample_point_within()
        z = self.sector.elevation_map.get_height_at(x, y)
    
        yaw = torch.randint(0, 360, (1,)).item()
        roll = torch.randint(0, 360, (1,)).item()
        pitch = torch.randint(0, 360, (1,)).item()
        return [x, y, z, yaw, roll, pitch]

    def render_from_camera(self, pose):
        assert len(pose) == 6 
        x, y, z, yaw, roll, pitch = pose

        self.camera.translate(x, y, z)
        self.camera.rotate(yaw, roll, pitch)
        rendering = render(self.camera, self.gaussian_model, self.background, device=self.device)
        return rendering

    def build_observation(self, agent_pose, goal_pose):
        image = self.render_from_camera(self.camera, agent_pose)  # you keep one shared camera instance
        aux = np.array([
            *agent_pose[:2],
            *goal_pose[:2],
            np.linalg.norm(np.array(agent_pose[:2]) - np.array(goal_pose[:2]))
        ], dtype=np.float32)
        return {"obs": image.cpu().numpy(), "aux": aux}

    