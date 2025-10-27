"""
USDZ Loader Module

Handles loading and accessing USDZ scene files for IsaacSim.
"""

from typing import Optional
import os


class USDZLoader:
    """
    USDZ scene data loader for a sector.

    Provides access to USDZ files that represent the 3D scene for IsaacSim.
    USDZ is a Universal Scene Description format used by IsaacSim.
    """

    def __init__(self, seq_id: str, sector_id: str, data_root: Optional[str] = None):
        """
        Initialize USDZLoader.

        Args:
            seq_id: Sequence identifier
            sector_id: Sector identifier
            data_root: Root directory for data files (uses default if None)
        """
        self.seq_id = seq_id
        self.sector_id = sector_id
        self.data_root = data_root or self._get_default_data_root()
        self._usdz_path: Optional[str] = None
        self._is_loaded = False

    def _get_default_data_root(self) -> str:
        """
        Get default data root path.

        Returns:
            Default data root directory
        """
        # TODO: Configure this through config file
        return "/path/to/data/root"

    def get_path(self) -> str:
        """
        Get path to USDZ file.

        Returns:
            Path to USDZ file

        Raises:
            FileNotFoundError: If USDZ file not found
        """
        if self._usdz_path is None:
            usdz_path = f"{self.data_root}/{self.seq_id}/{self.sector_id}/scene.usdz"
            if not os.path.exists(usdz_path):
                # Try alternative extension
                usdz_path = f"{self.data_root}/{self.seq_id}/{self.sector_id}/scene.usda"
            if not os.path.exists(usdz_path):
                raise FileNotFoundError(f"USDZ file not found for {self.seq_id}/{self.sector_id}")
            self._usdz_path = usdz_path
        return self._usdz_path

    def load(self) -> str:
        """
        Load USDZ file and return path.

        For USDZ files, loading typically means validating the file exists
        and returning the path for IsaacSim to load.

        Returns:
            Path to USDZ file
        """
        path = self.get_path()
        self._is_loaded = True
        return path

    @property
    def path(self) -> str:
        """
        Get USDZ file path.

        Returns:
            Path to USDZ file
        """
        return self.get_path()

    @property
    def is_loaded(self) -> bool:
        """
        Check if USDZ has been loaded.

        Returns:
            True if loaded, False otherwise
        """
        return self._is_loaded

    def exists(self) -> bool:
        """
        Check if USDZ file exists.

        Returns:
            True if file exists, False otherwise
        """
        try:
            self.get_path()
            return True
        except FileNotFoundError:
            return False

    def unload(self) -> None:
        """
        Unload USDZ data to free memory.
        """
        self._is_loaded = False

    def __repr__(self) -> str:
        """String representation."""
        return f"USDZLoader({self.seq_id}/{self.sector_id})"
