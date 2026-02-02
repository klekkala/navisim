"""RocksDB manager for storing USD assets and metadata.

This module provides a singleton RocksDB interface for storing:
- USD file paths for scene sections
- Section metadata (bounds, centers, etc.)
- Gaussian Splatting models (future)
- Spatial data references
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import threading
import atexit

from rocksdict import Rdict, Options

logger = logging.getLogger(__name__)


class RocksDBManager:
    """Thread-safe RocksDB manager for NaviSim Isaac Lab assets.

    Stores USD file paths, section metadata, and spatial data references.
    Compatible with NaviSim's original RocksDB structure but adapted for Isaac Sim.
    """

    def __init__(self, db_path: Optional[Path] = None, options: Optional[Options] = None):
        """Initialize RocksDB manager.

        Args:
            db_path: Path to RocksDB directory. Defaults to project_root/assets/rocksdb
            options: RocksDB options. Uses defaults if None.
        """
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
        """Get the default database path relative to project root.

        Looks for setup.py or pyproject.toml to identify project root,
        then returns assets/rocksdb under that root.

        Returns:
            Path to default RocksDB directory
        """
        # Find project root by looking for setup.py or pyproject.toml
        current = Path(__file__).resolve()

        # Walk up the directory tree to find project root
        for parent in current.parents:
            if (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
                return parent / "assets" / "rocksdb"

        # Fallback: assume we're in navisim_isaacsim package, go up to navisim root
        # navisim_isaacsim/navisim_lab/data/rocksdb_manager.py -> navisim/
        project_root = current.parents[3]
        return project_root / "assets" / "rocksdb"

    @staticmethod
    def get_project_root() -> Path:
        """Get the project root directory.

        Returns:
            Path to project root (where setup.py or pyproject.toml exists)
        """
        current = Path(__file__).resolve()

        # Look for project markers
        for parent in current.parents:
            if (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
                return parent

        # Fallback
        return current.parents[3]

    def _initialize(self):
        """Initialize RocksDB connection."""
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._db = Rdict(str(self.db_path), self.options)
            logger.info(f"RocksDB initialized at: {self.db_path}")
        except OSError as e:
            if "LOCK" in str(e):
                self._handle_lock_file()
                self._db = Rdict(str(self.db_path), self.options)
            else:
                raise

    def _handle_lock_file(self):
        """Handle stale lock file."""
        lock_file = self.db_path / "LOCK"
        if lock_file.exists():
            lock_file.unlink()
            logger.info(f"Removed stale lock file: {lock_file}")

    def save(self, key: str, value: Any):
        """Save a key-value pair to RocksDB.

        Args:
            key: String key
            value: Value to store (will be JSON serialized if dict/list)
        """
        with self._lock:
            # Serialize dicts and lists as JSON
            if isinstance(value, (dict, list)):
                value = json.dumps(value).encode("utf-8")
            elif isinstance(value, str):
                value = value.encode("utf-8")
            elif not isinstance(value, bytes):
                value = str(value).encode("utf-8")

            self._db[key] = value

    def get(self, key: str, default: Any = None, deserialize_json: bool = True) -> Any:
        """Get a value from RocksDB.

        Args:
            key: String key
            default: Default value if key not found
            deserialize_json: Try to deserialize as JSON if True

        Returns:
            Stored value, deserialized if JSON

        Raises:
            KeyError: If key not found and default is None
        """
        with self._lock:
            if default is None:
                if key not in self._db:
                    raise KeyError(f"Key not found: {key}")
                raw_value = self._db[key]
            else:
                raw_value = self._db.get(key, None)
                if raw_value is None:
                    return default

            # Deserialize bytes
            if isinstance(raw_value, bytes):
                raw_value = raw_value.decode("utf-8")

            # Try JSON deserialization
            if deserialize_json and isinstance(raw_value, str):
                try:
                    return json.loads(raw_value)
                except (json.JSONDecodeError, TypeError):
                    pass

            return raw_value

    def exists(self, key: str) -> bool:
        """Check if a key exists in the database.

        Args:
            key: String key

        Returns:
            True if key exists
        """
        with self._lock:
            return key in self._db

    def delete(self, key: str):
        """Delete a key from the database.

        Args:
            key: String key to delete
        """
        with self._lock:
            if key in self._db:
                del self._db[key]

    def save_section_metadata(
        self,
        section_id: str,
        usd_path: str,
        bounds: list,
        center: list,
        metadata: Optional[Dict] = None,
    ):
        """Save section metadata to RocksDB.

        Args:
            section_id: Unique section identifier
            usd_path: Path to USD file (relative or absolute)
            bounds: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            center: [x, y, z]
            metadata: Optional additional metadata
        """
        key = f"section:{section_id}"
        value = {
            "section_id": section_id,
            "usd_path": usd_path,
            "bounds": bounds,
            "center": center,
            "metadata": metadata or {},
        }
        self.save(key, value)
        logger.debug(f"Saved section metadata: {section_id}")

    def get_section_metadata(self, section_id: str) -> Optional[Dict]:
        """Get section metadata from RocksDB.

        Args:
            section_id: Section identifier

        Returns:
            Section metadata dict or None if not found
        """
        key = f"section:{section_id}"
        return self.get(key, default=None)

    def list_sections(self) -> list[str]:
        """List all section IDs in the database.

        Returns:
            List of section IDs
        """
        sections = []
        with self._lock:
            for key in self._db.keys():
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                if key.startswith("section:"):
                    sections.append(key.replace("section:", ""))
        return sections

    def save_sequence_graph_metadata(
        self,
        sequence_id: str,
        sectors: list[str],
        metadata: Optional[Dict] = None,
    ):
        """Save sequence graph metadata.

        Args:
            sequence_id: Unique sequence identifier
            sectors: List of sector/section IDs in this sequence
            metadata: Optional metadata
        """
        key = f"sequence:{sequence_id}"
        value = {
            "sequence_id": sequence_id,
            "sectors": sectors,
            "metadata": metadata or {},
        }
        self.save(key, value)
        logger.debug(f"Saved sequence metadata: {sequence_id}")

    def get_sequence_metadata(self, sequence_id: str) -> Optional[Dict]:
        """Get sequence metadata from RocksDB.

        Args:
            sequence_id: Sequence identifier

        Returns:
            Sequence metadata dict or None if not found
        """
        key = f"sequence:{sequence_id}"
        return self.get(key, default=None)

    def close(self):
        """Close the RocksDB connection."""
        if self._db:
            self._db.close()
            self._db = None
            logger.info(f"RocksDB closed: {self.db_path}")

    def __repr__(self) -> str:
        return f"RocksDBManager(path={self.db_path}, open={self._db is not None})"


# Global singleton instance
_db_instance: Optional[RocksDBManager] = None
_instance_lock = threading.Lock()


def get_db() -> RocksDBManager:
    """Get the global RocksDB instance (singleton pattern).

    Returns:
        Global RocksDBManager instance
    """
    global _db_instance

    if _db_instance is None:
        with _instance_lock:
            if _db_instance is None:
                _db_instance = RocksDBManager()

    return _db_instance


def reset_db():
    """Reset the global database instance (for testing/environment resets).

    This closes the current connection and clears the singleton.
    Next call to get_db() will create a fresh instance.
    """
    global _db_instance

    with _instance_lock:
        if _db_instance is not None:
            _db_instance.close()
            _db_instance = None
            logger.info("RocksDB instance reset")
