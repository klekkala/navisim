"""Utilities for capturing and saving camera images in Isaac Lab environments."""

import numpy as np
import torch
from pathlib import Path
from typing import Optional


def get_camera_images(env, camera_name: str = "jetbot_camera") -> Optional[torch.Tensor]:
    """Get RGB images from a camera sensor in the environment.

    Args:
        env: Isaac Lab environment (DirectRLEnv) with camera in scene.
             Use env.unwrapped to get past RSL-RL wrappers.
        camera_name: Name of camera in scene config.

    Returns:
        RGB tensor of shape (num_envs, H, W, 3) or None if not available.
    """
    scene = getattr(env, 'scene', None)
    if scene is None:
        return None

    try:
        camera = scene[camera_name]
    except (KeyError, IndexError):
        return None

    dt = getattr(env, 'physics_dt', 1.0 / 60.0)
    camera.update(dt=dt)

    if "rgb" in camera.data.output:
        return camera.data.output["rgb"]
    return None


def save_camera_image(camera_data: torch.Tensor, output_path: Path):
    """Save a single camera RGB tensor to an image file.

    Args:
        camera_data: RGB tensor, shape (H, W, 3) or (N, H, W, 3).
                     Values can be float [0,1] or uint8 [0,255].
        output_path: File path to save the image.
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL not found. Install with: pip install Pillow")
        return

    if camera_data.ndim == 4:
        camera_data = camera_data[0]

    img = camera_data.cpu().numpy() if isinstance(camera_data, torch.Tensor) else np.array(camera_data)

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(output_path)


def save_camera_batch(
    env,
    output_dir: Path,
    camera_name: str = "jetbot_camera",
    prefix: str = "frame",
    step: int = 0,
):
    """Save camera images from all environments to disk.

    Args:
        env: Isaac Lab environment (DirectRLEnv).
        output_dir: Directory to save images.
        camera_name: Name of camera in scene config.
        prefix: Filename prefix.
        step: Current step number for the filename.
    """
    rgb = get_camera_images(env, camera_name)
    if rgb is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for env_id in range(rgb.shape[0]):
        filename = f"{prefix}_step{step:06d}_env{env_id:02d}.png"
        save_camera_image(rgb[env_id], output_dir / filename)
