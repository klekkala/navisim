from data.rocksdb import RocksDB

import open3d as o3d
import json

class GaussianSplatting:
    def __init__(self, seq_id, sector_id):
        self.db = RocksDB()
        self.model_path = self.get_gaussian_splatting(seq_id, sector_id)
    
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
        
        return file_name
        