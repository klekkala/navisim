import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', '..')))


from config.gaussian_model_param import GaussianModelParam
from rendering.navisim_camera import NavisimCamera
from navisim.world.sector import Sector
from navisim.agents.pipeline import CustomPipeline

from third_party.scene import Scene
from third_party.gaussian_renderer import render
from third_party.gaussian_renderer import GaussianModel

import numpy as np
import torch

class NavisimScene(Scene):
    def __init__(self, model_params, camera : NavisimCamera, gaussian_model : GaussianModel, sector : Sector, background, device="cuda"):
        super().__init__(
            args=model_params,
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
    def create(cls, model_params: GaussianModelParam, sector : Sector, camera: NavisimCamera) -> "NavisimScene":
        """
        Factory method to create a NavisimScene instance using GaussianModelParam.

        Args:
            model_params (GaussianModelParam): Model config with path, SH degree, device, etc.

        Returns:
            NavisimScene: Initialized scene with loaded Gaussian model.
        """
        device = model_params.device
        gaussian_model = GaussianModel(sh_degree=model_params.sh_degree)
        background = torch.tensor(model_params.bg_color, dtype=model_params.dtype, device=device)
        return cls(model_params, camera, gaussian_model, sector, background, device)

    #TODO: needs to discuss how to determine the start pose
    def random_start_location(self):
        """
        Generate a random starting pose within the scene.

        Returns:
            list: A list containing x, y, z, yaw, roll, pitch.
        """

        x, y = 0,0  # self.sector.boundary.sample_point_within()
        z = 0       # self.sector.elevation_map.get_height_at(x, y)
        yaw = 0
        roll = 0
        pitch = np.pi

        return [x, y, z, yaw, roll, pitch] #[x, y, z, yaw, roll, pitch]
    
    def get_target_locations(self):
        """
        Returns:
            list: A list of coordinates that are currently occupied by obstacles
        """
        return self.sector.occupancy_map.get_occupied_coordinates()

    def render_from_camera(self, pose):
        assert len(pose) == 6 
        x, y, z, yaw, roll, pitch = pose

        self.camera.translate(x, y, z)
        self.camera.rotate(yaw, roll, pitch)
        render_result = render(viewpoint_camera = self.camera, pc = self.gaussian_model, pipe = CustomPipeline, bg_color = self.background)
        
        rendered_image = render_result["render"]
        rendered_image = torch.clamp(rendered_image, 0, 1)

        # Convert tensor to 0-255 and to byte format on GPU, then transfer to CPU
        rendered_image = (rendered_image * 255).byte().cpu()

        rendered_image = rendered_image.permute(1, 2, 0).contiguous()  # Change from CxHxW to HxWxC
        return rendered_image

    