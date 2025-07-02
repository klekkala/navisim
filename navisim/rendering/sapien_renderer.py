import logging
import numpy as np
from typing import Optional, List, float

try:
    from ...third_party.sapien import core as sapien
except ImportError:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from third_party.sapien import core as sapien

engine = sapien.Engine()
renderer = sapien.SapienRenderer(engine)
engine.set_renderer(renderer)

scene = None
logger = logging.getLogger(__name__)

def create_scene():
    scene = engine.create_scene()
    return scene

def get_scene():
    if scene is None:
        scene = create_scene()
    return scene

def set_lights(ambient_light:Optional[List[float]], dir_light:Optional[List[float]], point_lights:Optional[List[List[float]]]):
    """
    Set the lighting for the renderer.
    
    :param ambient_light: Ambient light color as [r, g, b, intensity]
    :param dir_light: Directional light as [[x, y, z], [r, g, b, intensity]]
    :param point_light: Point light as [[x, y, z], [r, g, b, intensity]]
    """
    if ambient_light:
        scene.set_ambient_light(ambient_light)
    
    if dir_light:
        scene.set_directional_light(dir_light[0], dir_light[1])
    
    if point_light:
        for point_light in point_lights:
            if len(point_light) == 2:
                scene.set_point_light(point_light[0], point_light[1])    

def set_camera(name: str, width: int=640, height: int=480, fovy = np.deg2rad(35), near:float = 0.1, far:float = 100):
    camera = scene.add_camera(
        name= name,
        width = width,
        height=height,
        fovy=fovy,
        near=near,
        far=far
    )
    return camera


