"""Utilities for working with cameras in Isaac Lab environments."""

import numpy as np
import torch
from pathlib import Path
from typing import Optional


def save_camera_image(
    camera_data: torch.Tensor,
    output_path: Path,
    normalize: bool = True
):
    """Save camera RGB data to image file.

    Args:
        camera_data: Camera RGB tensor, shape (H, W, 3) or (N, H, W, 3)
        output_path: Where to save the image
        normalize: Whether to normalize from [0, 1] to [0, 255]
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL not found. Install with: pip install Pillow")
        return

    # Handle batch dimension
    if camera_data.ndim == 4:
        camera_data = camera_data[0]  # Take first image

    # Convert to numpy and move to CPU
    if isinstance(camera_data, torch.Tensor):
        img = camera_data.cpu().numpy()
    else:
        img = np.array(camera_data)

    # Normalize if needed
    if normalize and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # Save image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(output_path)


def get_camera_images(env, camera_name: str = "jetbot_camera") -> Optional[torch.Tensor]:
    """Get RGB images from camera in environment.

    Args:
        env: Isaac Lab environment with camera in scene
        camera_name: Name of camera in scene (default: "jetbot_camera")

    Returns:
        RGB tensor of shape (num_envs, H, W, 3) or None if camera not found
    """
    if camera_name not in env.scene:
        print(f"Camera '{camera_name}' not found in scene")
        print(f"Available assets: {list(env.scene.keys())}")
        return None

    camera = env.scene[camera_name]

    # Update camera data
    camera.update(dt=env.physics_dt)

    # Get RGB data
    if "rgb" in camera.data.output:
        rgb = camera.data.output["rgb"]
        return rgb
    else:
        print(f"RGB data not available. Camera outputs: {list(camera.data.output.keys())}")
        return None


def visualize_camera_stream(
    env,
    camera_name: str = "jetbot_camera",
    window_name: str = "Jetbot POV",
    display_env_id: int = 0
):
    """Display camera stream in a window using OpenCV.

    Args:
        env: Isaac Lab environment
        camera_name: Name of camera in scene
        window_name: Name for display window
        display_env_id: Which environment to display (for multi-env)

    Returns:
        True if frame was displayed, False if window was closed or error
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not found. Install with: pip install opencv-python")
        return False

    # Get camera images
    rgb = get_camera_images(env, camera_name)
    if rgb is None:
        return False

    # Select environment to display
    if rgb.shape[0] > display_env_id:
        img = rgb[display_env_id].cpu().numpy()
    else:
        img = rgb[0].cpu().numpy()

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Display
    cv2.imshow(window_name, img_bgr)

    # Wait for key press (1ms) - allows window to update
    key = cv2.waitKey(1)

    # Return False if user pressed 'q' or closed window
    if key == ord('q'):
        cv2.destroyAllWindows()
        return False

    return True


def save_camera_batch(
    env,
    output_dir: Path,
    camera_name: str = "jetbot_camera",
    prefix: str = "frame",
    step: int = 0
):
    """Save camera images from all environments.

    Args:
        env: Isaac Lab environment
        output_dir: Directory to save images
        camera_name: Name of camera in scene
        prefix: Prefix for image filenames
        step: Current step number (for filename)
    """
    rgb = get_camera_images(env, camera_name)
    if rgb is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save image for each environment
    for env_id in range(rgb.shape[0]):
        img = rgb[env_id]
        filename = f"{prefix}_step{step:06d}_env{env_id:02d}.png"
        save_camera_image(img, output_dir / filename)

    print(f"Saved {rgb.shape[0]} images to {output_dir}")
