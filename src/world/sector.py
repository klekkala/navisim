from collections import defaultdict
from rendering.gaussian_splatting import GaussianSplatting
from spaces.elevation_map import ElevationMap
from spaces.occupancy_map import OccupancyMap
from spaces.boundary_polygon import BoundaryPolygon

class Sector:
    def __init__(self, seq_id, sector_id):
        self.seq_id = seq_id
        self.sector_id = f'sector{sector_id}'
        self.elevation_map = ElevationMap(seq_id, self.sector_id)
        self.occupancy_map = OccupancyMap(seq_id, self.sector_id)
        self.boundary = BoundaryPolygon(seq_id, self.sector_id)
        self.gaussian_model = GaussianSplatting(seq_id, self.sector_id)

        self.prev = None
        self.next = None

    def __repr__(self):
        return f"Sector {self.sector_id}"
        
        
    
    