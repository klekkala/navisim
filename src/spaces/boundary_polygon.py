from data.rocksdb import RocksDB
from shapely import wkb
from shapely.geometry import Point
from enum import Enum, auto

import math
import json

class RelativeDir(Enum):
    INSIDE_OR_ON = auto()
    OUTSIDE_TOP = auto()
    OUTSIDE_RIGHT = auto()
    OUTSIDE_BOTTOM = auto()
    OUTSIDE_LEFT = auto()

class BoundaryPolygon:
    def __init__(self, seq_id, sector_id):
        self.db = RocksDB()
        self.boundary_polygon = self._get_polygon(seq_id, sector_id)
    
    def __getattr__(self, name):
        return getattr(self.boundary_polygon, name)

    def contains(self, x: float, y: float) -> bool:
        return self.polygon.covers(Point(x, y))

    def direction_to(self, x: float, y: float) -> RelativeDir:
        point = Point(x, y)
        if self.polygon.covers(point):
            return RelativeDir.INSIDE_OR_ON

        centroid = self.polygon.centroid
        dx = x - centroid.x
        dy = centroid.y - y  # Flip Y axis for clock direction

        angle = math.degrees(math.atan2(dy, dx)) % 360

        if 45 <= angle < 135:
            return RelativeDir.OUTSIDE_TOP
        elif 135 <= angle < 225:
            return RelativeDir.OUTSIDE_LEFT
        elif 225 <= angle < 315:
            return RelativeDir.OUTSIDE_BOTTOM
        else:
            return RelativeDir.OUTSIDE_RIGHT
    
    def _get_polygon(self, seq_id, sector_id):
        date, session = seq_id.split('/')

        key = json.dumps({
                "date": date,
                "session": session,
                "sector": str(sector_id),
                "file_name": "boundary"
            })
        
        data = self.db.get(key)
        polygon = wkb.loads(data)

        return polygon
