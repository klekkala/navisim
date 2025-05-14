from data.rocksdb import RocksDB
from spaces.boundary_polygon import BoundaryPolygon
from scipy.ndimage import map_coordinates

import json
import numpy as np
import io


class ElevationMap: 
    def __init__(self, seq_id, sector_id):
        self.db = RocksDB()
        self.map = self._get_elevation_map(seq_id, sector_id)

    def get_height_at(self, x: float, y: float) -> float:
        coords = np.array([[x], [y]])
        if 0 <= x < self.map.shape[0] and 0 <= y < self.map.shape[1]:
            return float(map_coordinates(self.map, coords, order=1, mode='nearest')[0])
        else:
            return 0.0

    def _get_elevation_map(self, seq_id, sector_id):
        try:
            date, session = seq_id.split('/')
            key = json.dumps({
                "date": date,
                "session": session,
                "sector": str(sector_id),
                "file_name": "elevation"
            })

            raw_data = self.db.get(key)
            buffer = io.BytesIO(raw_data)
            elevation_map =  np.load(buffer, allow_pickle=True)
            return elevation_map
        
        except EOFError as e:
            print(f"Error loading elevation map for {seq_id}, {sector_id}: {e}")
            return None
    
    def __getitem__(self, index):
        return self.map[index]