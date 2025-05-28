from config.gaussian_model_param import GaussianModelParam
from scene.navisim_camera import NavisimCamera
from submodules.gaussian.scene import Scene
from submodules.gaussian.gaussian_renderer import render
from submodules.gaussian.gaussian_renderer import GaussianModel


import torch

class NavisimScene(Scene):
    def __init__(self, model_params, gaussian_model, background, device="cuda"):
        super().__init__(
            model_params=model_params,
            gaussians=gaussian_model,
            load_iteration=-1,
            shuffle=False
        )
        self.gaussian_model = gaussian_model
        self.background = background
        self.device = device

    @classmethod
    def create(cls, model_params: GaussianModelParam) -> "NavisimScene":
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
        return cls(resolved_params, gaussian_model, background, device)

    def render_from_camera(self, camera : NavisimCamera, pose):
        assert len(pose) == 6 
        x, y, z, yaw, roll, pitch = pose

        camera.translate(x, y, z)
        camera.rotate(yaw, roll, pitch)
        
        rendering = render(camera, self.gaussian_model, self.background, device=self.device)
        return rendering

    