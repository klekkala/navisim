from data.rocksdb import RocksDB

import open3d as o3d
import json

class GaussianSplatting:
    def __init__(self, seq_id, sector_id):
        self.db = RocksDB()
        self.gs = self.get_gaussian_splatting(seq_id, sector_id, sector_id)
    
    def get_gaussian_splatting(self, seq_id, sector_id):
        date, session = seq_id.split('/')
        key = json.dumps({
            "date": date,
            'session': session,
            'sector': sector_id,
            'file_name': 'gaussian'
            }, sort_keys=True)
        data  = self.db.get(key)
        file_name = f'{date}_{session}_{sector_id}.ply'
        with open(file_name, "wb") as f:
            f.write(data)

        try:
            pcd = o3d.io.read_point_cloud(file_name)
            if len(pcd.points) == 0:
                    print("❌ Loaded PLY file, but contains no points.")
            else:
                print(f"✅ Loaded PLY file with {len(pcd.points)} points.")
            return pcd
        except Exception as e:
            print("❌ Failed to load Gaussian splat:", e)
        