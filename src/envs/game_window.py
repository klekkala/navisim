import pygame
import numpy as np
from torchvision.transforms import ToPILImage

class GameWindow:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
  
        # Set up window dimensions (width, height)
        self.image1_width = (1959 / 2)
        self.image1_height = (1090 / 2)
        self.image2_width = 600
        self.image2_height = 600

        self.image_scale()

        # Set up the window
        self.window_width = self.image1_width + self.image2_width
        self.window_height = self.image1_height

        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        
        # Fill the window with a color (black here)
        self.window.fill((0, 0, 0))
        pygame.display.set_caption('Gaussian Splatting with Elevation Map')
        
        # To track if the window is running
        self.running = True

    # priorizing image1's sizes
    def image_scale(self):
        def helper(width, max_width, height, max_height):
            width_ratio = max_width / width
            height_ratio = max_height / height
            scale_factor = min(width_ratio, height_ratio)

            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return new_width, new_height

        self.image1_width, self.image1_height = helper(
            self.image1_width, self.image1_width, 
            self.image1_height, self.image1_height
        )

        self.image2_width, self.image2_height = helper(
            self.image2_width, self.image2_width, 
            self.image2_height, self.image1_height
        )

    def quit(self):
        pygame.quit()

    def tensor_to_surface(self, tensor):
        # Reshape and prepare tensor for Pygame
        h, w = tensor.shape[0], tensor.shape[1]
        return pygame.image.fromstring(tensor.numpy().tobytes(), (w, h), 'RGB')

    # Function to display images on demand
    def display_images(self, image1, image2):
        image1 = self.tensor_to_surface(image1)
        image1 = pygame.transform.scale(image1, (self.image1_width, self.window_height))

        if isinstance(image2, np.ndarray):
            image2 = self.numpy_to_surface(image2)
        else:
            image2 = pygame.image.load(image2)

        image2 = pygame.transform.scale(image2, (self.image2_width, self.window_height))

        self.window.blit(image1, (0, 0))
        self.window.blit(image2, (self.image1_width, 0))
        pygame.display.flip()
    
    def numpy_to_surface(self, elevation_map: np.ndarray) -> pygame.Surface:
        # Normalize
        normalized = (elevation_map - np.nanmin(elevation_map)) / (np.nanmax(elevation_map) - np.nanmin(elevation_map))
        normalized = np.nan_to_num(normalized)
        grayscale = (normalized * 255).astype(np.uint8)

        # Stack to RGB
        rgb = np.stack((grayscale,) * 3, axis=-1)

        # Convert to Surface (transpose to (W, H, 3))
        surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        return surface