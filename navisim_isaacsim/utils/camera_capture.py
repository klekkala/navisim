# utils/camera_capture.py

"""Utility functions for capturing images from Isaac Sim cameras."""

import os
from datetime import datetime
import numpy as np
from PIL import Image


def save_camera_image(rgb_data: np.ndarray, output_dir: str = "outputs/snapshots", prefix: str = "camera"):
    """Save camera RGB data as an image file.

    Args:
        rgb_data: RGB image data as numpy array (H, W, 3) or (H, W, 4) with alpha
        output_dir: Directory to save images
        prefix: Prefix for the filename

    Returns:
        str: Path to saved image file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # Convert to uint8 if needed
    if rgb_data.dtype == np.float32 or rgb_data.dtype == np.float64:
        rgb_data = (rgb_data * 255).astype(np.uint8)

    # Remove alpha channel if present
    if rgb_data.shape[-1] == 4:
        rgb_data = rgb_data[:, :, :3]

    # Save image
    img = Image.fromarray(rgb_data)
    img.save(filepath)

    return filepath


def get_camera_from_prim(sim, camera_prim_path: str):
    """Get camera data from a USD prim path.

    Args:
        sim: Simulation context
        camera_prim_path: Path to camera prim (e.g., "/World/envs/env_0/Jetbot/chassis/front_cam")

    Returns:
        Camera object that can be used to capture images
    """
    from omni.isaac.core.prims import Camera

    camera = Camera(prim_path=camera_prim_path)
    return camera


def capture_camera_snapshot(camera, output_dir: str = "outputs/snapshots", prefix: str = "camera"):
    """Capture and save a snapshot from an Isaac Sim camera.

    Args:
        camera: Camera object (from get_camera_from_prim or Camera sensor)
        output_dir: Directory to save images
        prefix: Prefix for the filename

    Returns:
        str: Path to saved image file
    """
    # Get current frame from camera
    rgb_data = camera.get_rgba()  # Returns numpy array

    # Save the image
    filepath = save_camera_image(rgb_data, output_dir, prefix)

    print(f"[Camera] Snapshot saved to: {filepath}")
    return filepath
