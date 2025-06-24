from navisim.data.rocksdb import RocksDB
from plyfile import PlyData

from io import BytesIO
from pathlib import Path
import json

class GaussianSplatting:
    def __init__(self, seq_id, sector_id):
        self.db = RocksDB()
        self.model_path = self.get_gaussian_splatting(seq_id, sector_id)
    
    def get_gaussian_splatting(self, seq_id, sector_id) -> str: 
        date, session = seq_id.split('/')
        base_path = Path("gs", date, session, str(sector_id))
        ply_path = base_path / "point_cloud" / "iteration_0" / "point_cloud.ply"

        # Skip if file already exists
        if ply_path.exists():
            return str(base_path)

        # Otherwise, fetch from RocksDB and save
        key = json.dumps({
            "date": date,
            "session": session,
            "sector": sector_id,
            "file_name": "gaussian"
        })

        data = self.db.get(key)
        plydata = PlyData.read(BytesIO(data))

        ply_path.parent.mkdir(parents=True, exist_ok=True)

        with open(ply_path, "wb") as f:
            plydata.write(f)

        return str(base_path)
        