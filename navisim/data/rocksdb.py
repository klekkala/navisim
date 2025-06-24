from rocksdict import Rdict, Options
import threading
import atexit
import numpy as np

class RocksDB:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path="/lab/kiran/navisim-1/assets/rocksdb", options=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(RocksDB, cls).__new__(cls)
                    cls._instance._initialize(db_path, options)
                    atexit.register(cls._instance.close)  # ✅ Ensure close on exit
        return cls._instance
    
    def __getattr__(self, name):
        # Avoid recursion if 'db' isn't initialized yet
        if name == "db" or not hasattr(self, "db"):
            raise AttributeError(f"'RocksDB' object has no attribute '{name}'")
        return getattr(self.db, name)

    def _initialize(self, db_path, options):
        options = options if options else self._get_options()
        self.db_path = db_path
        try:
            self.db = Rdict(db_path, options)
        except OSError as e:
            if "LOCK" in str(e):
                print("[WARN] Detected lock file. Attempting to clean up...")
                import os
                lock_file = os.path.join(db_path, "LOCK")
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                # Retry once after removal
                self.db = Rdict(db_path, options)
            else:
                raise
    
    def _get_options(self):
        return Options()

    def save(self, key, value):
        self.db[key] = value

    def get(self, key):
        if key not in self.db:
            raise KeyError(f"Key not found in RocksDB: {key}")
        return self.db.get(key)

    def close(self):
        if hasattr(self, "db"):
            self.db.close()
            del self.db
        RocksDB._instance = None  
    
    def delete(self):
        self.db.close()
        Rdict.destroy(self.db_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


    @classmethod
    def reset_instance(cls):
        if cls._instance:
            cls._instance.close()
            cls._instance = None