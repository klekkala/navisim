from collections import defaultdict
from navisim.rendering.gaussian_splatting import GaussianSplatting
from navisim.spaces.elevation_map import ElevationMap
from navisim.spaces.occupancy_map import OccupancyMap
from navisim.spaces.boundary_polygon import BoundaryPolygon

class Sector:
    def __init__(self, seq_id, sector_id):
        self.seq_id = seq_id
        self.sector_id = f'sector{sector_id}'

        self._elevation_map = None
        self._occupancy_map = None
        self._boundary = None
        self._gaussian_model = None

        self.prev = None
        self.next = None

        self.prev = None
        self.next = None

    @property
    def elevation_map(self):
        if self._elevation_map is None:
            print(f"Loading ElevationMap for {self.seq_id}/{self.sector_id}")
            self._elevation_map = ElevationMap(self.seq_id, self.sector_id)
        return self._elevation_map

    @property
    def occupancy_map(self):
        if self._occupancy_map is None:
            print(f"Loading OccupancyMap for {self.seq_id}/{self.sector_id}")
            self._occupancy_map = OccupancyMap(self.seq_id, self.sector_id)
        return self._occupancy_map

    @property
    def boundary(self):
        if self._boundary is None:
            print(f"Loading BoundaryPolygon for {self.seq_id}/{self.sector_id}")
            self._boundary = BoundaryPolygon(self.seq_id, self.sector_id)
        return self._boundary

    @property
    def gaussian_model(self):
        if self._gaussian_model is None:
            print(f"Loading GaussianSplatting for {self.seq_id}/{self.sector_id}")
            self._gaussian_model = GaussianSplatting(self.seq_id, self.sector_id)
        return self._gaussian_model
    
    def __repr__(self):
        return f"Sector {self.sector_id}"
        

    def unload_all(self):
        """Frees memory for all metadata components."""
        self._elevation_map = None
        self._occupancy_map = None
        self._boundary = None
        self._gaussian_model = None
        
    
    