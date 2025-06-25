import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import json
import io

import sys
import os

from navisim.spaces.elevation_map import ElevationMap
from unittest.mock import Mock

@pytest.fixture
def mock_rocksdb(monkeypatch):
    mock_db = Mock()
    monkeypatch.setattr("navisim.spaces.elevation_map.RocksDB", lambda: mock_db)
    return mock_db

# Helper: Generate fake elevation data as bytes
def fake_elevation_bytes():
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=True)
    return buffer.getvalue(), arr

# Patch BoundaryPolygon globally
@pytest.fixture(autouse=True)
def patch_boundary_polygon(monkeypatch):
    monkeypatch.setattr("spaces.boundary_polygon.BoundaryPolygon", lambda seq_id, sector_id: "MockBoundary")

def test_elevation_map_loads_correctly(mock_rocksdb):
    from navisim.spaces.elevation_map import ElevationMap

    fake_bytes, expected_array = fake_elevation_bytes()
    mock_rocksdb.get.return_value = fake_bytes

    key = json.dumps({
        "date": "2023-01-01",
        "session": "session1",
        "sector": "sector5",
        "file_name": "elevation"
    })

    emap = ElevationMap("2023-01-01/session1", "sector5")

    np.testing.assert_array_equal(emap.map, expected_array)
    assert emap[0][0] == 1.0
    mock_rocksdb.get.assert_called_once_with(key)

def test_elevation_map_handles_missing_data(mock_rocksdb):
    from navisim.spaces.elevation_map import ElevationMap

    mock_rocksdb.get.return_value = None

    emap = ElevationMap("2023-01-01/session1", "sector5")

    assert emap.map is None

def test_elevation_map_handles_corrupt_data(mock_rocksdb):
    from navisim.spaces.elevation_map import ElevationMap

    mock_rocksdb.get.return_value = b"not a valid npy"

    emap = ElevationMap("2023-01-01/session1", "sector5")

    assert emap.map is None