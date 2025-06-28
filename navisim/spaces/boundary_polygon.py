import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from typing import Tuple
from shapely import wkb
from shapely.geometry import Point
from navisim.enums.enums import RelativeDir
from navisim.data.rocksdb import get_db

import math
import json

class BoundaryPolygon:
    def __init__(self, seq_id, sector_id):
        self.db = get_db()
        self.polygon = self._get_polygon(seq_id, sector_id)
    
    def __getattr__(self, name):
        return getattr(self.polygon, name)

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

    def sample_point_within(self) -> Tuple[float, float]:
        
        center_point = self.polygon.centroid
        return center_point.x, center_point.y

        #TODO change sampling method later
        """Uniformly sample a point within the polygon (using rejection sampling)."""
        from shapely.geometry import Polygon, Point
        import random
        # minx, miny, maxx, maxy = self.polygon.bounds

        # while True:
        #     x, y = random.uniform(minx, maxx), random.uniform(miny, maxy)
        #     if self.polygon.contains(Point(x, y)):
        #         return x, y