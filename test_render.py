import sys
import os
import time
import numpy as np

# Add the absolute path to the `navisim/` directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, project_root)

from navisim.rendering.navisim_camera import NavisimCamera
from navisim.config.gaussian_model_param import GaussianModelParam
from torchvision.utils import save_image
import torch

ply_path = '../Data/12a18dcf-1/point_cloud/iteration_30000/point_cloud.ply'
out_path = '../renders/'
num_frames = 100

from gaussian_splatting import scene, gaussian_renderer, utils

start_time = time.time()

GS_sectors = [scene.GaussianModel(3) for _ in range(5)]

for GS in GS_sectors:
    GS.load_ply(ply_path)

class CustomPipeline:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False

model_params=GaussianModelParam.create(model_path=ply_path)

camera = NavisimCamera.create(camera_id=str(9), W = 720,H = 480)
camera.image_height = camera.H
camera.image_width = camera.W

#camera.world

background = torch.tensor(model_params.bg_color, dtype=model_params.dtype, device=0).float()

start_time = time.time()
math_time = 0
render_time = 0

for i in range(num_frames):
    cycle_start = time.time()
    rots = np.random.rand(3)*np.pi*2
    trans = np.random.rand(3)*5
    camera.rotate(rots[0], rots[1], rots[2])
    camera.translate(trans[0], trans[1], trans[2])

    camera.camera_center = torch.from_numpy(camera.T).float().to(device="cuda:0")
    camera.world_view_transform = torch.from_numpy(utils.graphics_utils.getWorld2View(camera.R, camera.T)).float().to(device="cuda:0")
    camera.full_proj_transform = utils.graphics_utils.getProjectionMatrix(0.1, 1000, camera.FoVx, camera.FoVy).float().to(device="cuda:0") # ~670fps

    math_done = time.time()
    math_time += math_done - cycle_start

    render_result = gaussian_renderer.render(viewpoint_camera = camera, pc = GS_sectors[i%5], pipe = CustomPipeline, bg_color = background) # ~1200fps *Goal
    render_done = time.time()
    render_time += render_done - math_done

    #save_image(render_result["render"], out_path + 'img' + str(i) + '.png')  # ~10fps

end_time = time.time()

print(end_time - start_time)
print(math_time)
print(render_time)
print(num_frames/(end_time - start_time))
