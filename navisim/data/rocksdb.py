
from typing import Optional
from rocksdict import Rdict, Options
import threading
import atexit
import os

from pathlib import Path
from rocksdict import Rdict, Options
import threading
import atexit
import os

class RocksDBManager:
    def __init__(self, db_path: Optional[Path] = None, options=None):
        # Default to project_root/assets/rocksdb
        db_path = db_path or self.get_default_db_path()
        self.db_path = Path(db_path)
        self.options = options or Options()
        self._db = None
        self._lock = threading.Lock()
        self._initialize()
        atexit.register(self.close)
    
    @staticmethod
    def get_default_db_path() -> Path:
        """Get the default database path relative to project root"""
        # Find project root by looking for setup.py or pyproject.toml
        current = Path(__file__).resolve()
        
        # Walk up the directory tree to find project root
        for parent in current.parents:
            if (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
                return parent / "assets" / "rocksdb"
        
        # Fallback: assume we're in navisim package, go up two levels
        project_root = current.parents[2]  # navisim/database.py -> navisim-1/
        return project_root / "assets" / "rocksdb"
    
    @staticmethod
    def get_project_root() -> Path:
        """Get the project root directory"""
        current = Path(__file__).resolve()
        
        # Look for project markers
        for parent in current.parents:
            if (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
                return parent
                
        # Fallback
        return current.parents[2]
    
    def _initialize(self):
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self._db = Rdict(str(self.db_path), self.options)
            print(f"[INFO] RocksDB initialized at: {self.db_path}")
        except OSError as e:
            if "LOCK" in str(e):
                self._handle_lock_file()
                self._db = Rdict(str(self.db_path), self.options)
            else:
                raise
    
    def _handle_lock_file(self):
        lock_file = self.db_path / "LOCK"
        if lock_file.exists():
            lock_file.unlink()
            print(f"[INFO] Removed lock file: {lock_file}")
    
    def save(self, key, value):
        with self._lock:
            self._db[key] = value
    
    def get(self, key, default=None):
        with self._lock:
            if default is None:
                if key not in self._db:
                    raise KeyError(f"Key not found: {key}")
                return self._db[key]
            return self._db.get(key, default)
    
    def exists(self, key):
        with self._lock:
            return key in self._db
    
    def close(self):
        if self._db:
            self._db.close()
            self._db = None
            print(f"[INFO] RocksDB closed: {self.db_path}")

# Global instance
_db_instance = RocksDBManager()

def get_db() -> RocksDBManager:
    return _db_instance

def reset_db():
    """For testing"""
    _db_instance.close()
    _db_instance = None