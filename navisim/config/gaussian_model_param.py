import torch

from argparse import ArgumentParser
from pathlib import Path

try:
    from third_party.gaussian_splatting.arguments import ModelParams
except ImportError:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from third_party.gaussian_splatting.arguments import ModelParams


class GaussianModelParam(ModelParams):
    def __init__(self, parser):
        # Initialize default parameters
        self.model_path: str
        self.sh_degree: int = 3
        self.bg_color: list[float] = [0.0, 0.0, 0.0]
        self.dtype = torch.float32
        self.device: str = "cuda"
        super().__init__(parser, sentinel=True)

    @classmethod
    def create(
        cls,
        model_path: str,
        parser = None,
        sh_degree: int = 3,
        bg_color: list[float] = [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device: str = "cuda",
    ) -> "GaussianModelParam":
        """
        Factory method to create a GaussianModelParam with custom overrides.

        Args:
            parser: Argument parser or placeholder used by ModelParams.
            model_path (str): Path to the saved Gaussian model.
            sh_degree (int): Spherical harmonics degree.
            bg_color (list[float]): Background color.
            dtype: Torch dtype (default float32).
            device (str): Device to use (default 'cuda').

        Returns:
            GaussianModelParam: Configured parameter object.
        """
        if parser is None:
           parser = ArgumentParser()

        instance = cls(parser)
        instance.model_path = Path(model_path).resolve().as_posix()
        instance.sh_degree = sh_degree
        instance.bg_color = bg_color
        instance.dtype = dtype
        instance.device = device
        return instance

    def extract(self, args):
        # Optionally override this if needed
        return super().extract(args)
