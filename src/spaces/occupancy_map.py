import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.rocksdb import RocksDB
from spaces.boundary_polygon import BoundaryPolygon

import json
import numpy as np
import io

class OccupancyMap:
    def __init__(self, seq_id, sector_id):
        self.db = RocksDB()
        self.map = self._get_occupancy_map(seq_id, sector_id)
        self.boundary = BoundaryPolygon(seq_id, sector_id)

    def __getitem__(self, index):
        return self.map[index]
    
    def is_occupied(self):
        return False

    def _get_occupancy_map(self, seq_id, sector_id):
        try:
            date, session = seq_id.split('/')
            key = json.dumps({
                "date": date,
                "session": session,
                "sector": str(sector_id),
                "file_name": "occupancy"
            })
            
            raw_data = self.db.get(key)
            buffer = io.BytesIO(raw_data)
            occupancy_map =  np.load(buffer, allow_pickle=True)
            return occupancy_map
        except EOFError as e:
            print(f"Error loading occupancy map for {seq_id}, {sector_id}: {e}")
            return None
    
    def __getitem__(self, index):
        return self.map[index]