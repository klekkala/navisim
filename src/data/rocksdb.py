from rocksdict import Rdict, Options
import threading
import numpy as np

class RocksDB:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path="/lab/kiran/navisim-1/assets/rocksdb", options = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(RocksDB, cls).__new__(cls)
                    cls._instance._initialize(db_path, options)
        return cls._instance
    
    def __getattr__(self, name):
        return getattr(self.db, name)

    def _initialize(self, db_path, options):
        options = options if options else self._get_options()
        self.db_path = db_path
        self.db = Rdict(db_path, options)
    
    def _get_options(self):
        return Options()

    def save(self, key, value):
        self.db[key] = value

    def get(self, key):
        if key not in self.db:
            raise KeyError(f"Key not found in RocksDB: {key}")
        return self.db.get(key)

    def close(self):
        self.db.close()
    
    def delete(self):
        self.db.close()
        Rdict.destroy(self.db_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

