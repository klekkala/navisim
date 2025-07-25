import sys
import os
import time
import math
import numpy as np

# Add the absolute path to the `navisim/` directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, project_root)

gs_root = os.path.abspath(os.path.join(os.getcwd(), "third_party/gaussian_splatting"))
sys.path.insert(0, gs_root)

from third_party.gaussian_splatting import scene, gaussian_renderer
from third_party.gaussian_splatting.utils import graphics_utils
from diff_gaussian_rasterization import multithread_manager, GaussianRasterizationSettings

from navisim.render.navisim_camera import NavisimCamera
from navisim.config.gaussian_model_param import GaussianModelParam
from torchvision.utils import save_image
import torch

ply_path = '../Data/12a18dcf-1/point_cloud/iteration_30000/point_cloud.ply'
out_path = '../renders/'
num_frames = 100

start_time = time.time()

GS_sectors = [scene.GaussianModel(3) for _ in range(5)]
mt_manager = multithread_manager()

for GS in GS_sectors:
    GS.load_ply(ply_path)

class CustomPipeline:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False

model_params=GaussianModelParam.create(model_path=ply_path)

camera = NavisimCamera.create(camera_id=str(9), W = 720,H = 480)
camera.image_height = camera.H
camera.image_width = camera.W

#camera.world

background = torch.tensor(model_params.bg_color, dtype=model_params.dtype, device=0).float()

start_time = time.time()
math_time = 0
render_time = 0

for i in range(5):
    cycle_start = time.time()
    rots = np.random.rand(3)*np.pi*2
    trans = np.random.rand(3)*5
    camera.rotate(rots[0], rots[1], rots[2])
    camera.translate(trans[0], trans[1], trans[2])

    camera.camera_center = torch.from_numpy(camera.T).float().to(device="cuda:0")
    camera.world_view_transform = torch.from_numpy(graphics_utils.getWorld2View(camera.R, camera.T)).float().to(device="cuda:0")
    camera.full_proj_transform = graphics_utils.getProjectionMatrix(0.1, 1000, camera.FoVx, camera.FoVy).float().to(device="cuda:0") # ~670fps

    math_done = time.time()
    math_time += math_done - cycle_start

    #render_result = gaussian_renderer.render(viewpoint_camera = camera, pc = GS_sectors[i%5], pipe = CustomPipeline, bg_color = background) # ~1200fps *Goal
    pc = GS_sectors[i%5]
    
    means3D = pc.get_xyz
    means2D = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    opacity = pc.get_opacity
    dc, shs = pc.get_features_dc, pc.get_features_rest
    scales = pc.get_scaling
    rotations = pc.get_rotation

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx = math.tan(camera.FoVx * 0.5),
        tanfovy = math.tan(camera.FoVy * 0.5),
        bg=background,
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=CustomPipeline.debug,
        antialiasing=CustomPipeline.antialiasing
    )

    mt_manager.py_add_sector(
        ID = 42,
        means3D = means3D,
        means2D = means2D,
        sh = shs,
        colors_precomp = torch.tensor(0, device="cuda"), # None
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3Ds_precomp = torch.tensor(0, device="cuda"), # None
        raster_settings = raster_settings)
    
    render_done = time.time()
    render_time += render_done - math_done

    #save_image(render_result["render"], out_path + 'img' + str(i) + '.png')  # ~10fps

end_time = time.time()

print(end_time - start_time)
print(math_time)
print(render_time)
print(num_frames/(end_time - start_time))
