import os
import pygame
import numpy as np
import torch

class GameWindow:
    def __init__(self, width=1280, height=720):
        # Initialize Pygame
        pygame.init()


        # Calculate window size
        self.width = width #int(self.image1_width + self.image2_width)
        self.height = height #int(self.image1_height)
        
        # Set up image dimensions
        self.image2_width = 600
        self.image2_height = 600

        self.image_scale()


        # Determine if headless mode (i.e., Jupyter with dummy video driver)
        self.headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"

        # Use an off-screen surface in headless mode
        if self.headless:
            self.window = pygame.Surface((self.width, self.height))
        else:
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Gaussian Splatting with Elevation Map')

        self.window.fill((0, 0, 0))
        self.running = True

    def image_scale(self):
        def helper(width, max_width, height, max_height):
            width_ratio = max_width / width
            height_ratio = max_height / height
            scale_factor = min(width_ratio, height_ratio)
            return int(width * scale_factor), int(height * scale_factor)

        self.width, self.height = helper(
            self.width, self.width,
            self.height, self.height
        )
        self.image2_width, self.image2_height = helper(
            self.image2_width, self.image2_width,
            self.image2_height, self.height
        )

    def quit(self):
        pygame.quit()

    def tensor_to_surface(self, tensor: torch.Tensor) -> pygame.Surface:
        h, w = tensor.shape[0], tensor.shape[1]
        return pygame.image.fromstring(tensor.numpy().tobytes(), (w, h), 'RGB')


    def numpy_to_surface(self, elevation_map: np.ndarray) -> pygame.Surface:
        normalized = (elevation_map - np.nanmin(elevation_map)) / (np.nanmax(elevation_map) - np.nanmin(elevation_map) + 1e-8)
        normalized = np.nan_to_num(normalized)
        grayscale = (normalized * 255).astype(np.uint8)
        rgb = np.stack((grayscale,) * 3, axis=-1)
        surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        return surface

    def display_images(self, image1_tensor: torch.Tensor, image2_array: np.ndarray):
        image1_surface = self.tensor_to_surface(image1_tensor)
        # image2_surface = self.numpy_to_surface(image2_array)

        image1_surface = pygame.transform.scale(image1_surface, (self.width, self.height))
        # image2_surface = pygame.transform.scale(image2_surface, (self.image2_width, self.window_height))

        self.window.blit(image1_surface, (0, 0))
        # self.window.blit(image2_surface, (self.image1_width, 0))

        # Only flip if there's a display (not headless mode)
        if not self.headless:
            pygame.display.flip()

    def get_window_pixels(self):
        """Returns the current window surface as a numpy array (HWC RGB)."""
        pixels = pygame.surfarray.array3d(self.window)  # (W, H, 3)
        return np.transpose(pixels, (1, 0, 2))          # ➝ (H, W, 3)