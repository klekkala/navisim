# NaviSim Module

The NaviSim module is organized into nested folders to better represent component categories:

## Structure

```
navisim/
├── __init__.py           # Main module exports
├── converters.py         # Data format converters
├── world/                # Navigation graph structure
│   ├── __init__.py
│   ├── sequence_graph.py # NetworkX-based graph
│   └── sector.py         # Individual sectors
└── spaces/               # Spatial data components
    ├── __init__.py
    ├── height_field.py   # Height field data
    ├── usdz_loader.py    # USDZ scene loader
    ├── boundary_polygon.py # Boundary polygons
    └── elevation_map.py  # Elevation map generation
```

## Organization

### `world/` - Navigation Graph Structure
Contains the core navigation graph and sector management:
- **SequenceGraph**: Manages the NetworkX graph of sequences
- **Sector**: Represents individual sectors with lazy-loaded spatial data

### `spaces/` - Spatial Data Components
Contains spatial data structures used by sectors:
- **HeightField**: Terrain height field data (`.npy` format)
- **USDZLoader**: USDZ scene file management (`.usdz`/`.usda` format)
- **BoundaryPolygon**: Spatial boundary constraints (`.npy` format)
- **ElevationMapGenerator**: Generates elevation maps for IsaacSim

### Root Level
- **converters.py**: Utilities for converting data between NaviSim and IsaacSim formats

## Usage

All components can be imported directly from the main `navisim` module:

```python
from navisim_isaac.navisim import (
    SequenceGraph,
    Sector,
    HeightField,
    USDZLoader,
    BoundaryPolygon,
    ElevationMapGenerator,
    DataConverter
)
```

Or from specific submodules:

```python
from navisim_isaac.navisim.world import SequenceGraph, Sector
from navisim_isaac.navisim.spaces import HeightField, USDZLoader, BoundaryPolygon
from navisim_isaac.navisim.converters import DataConverter
```

## Data Flow

1. **SequenceGraph** loads the navigation graph from pickle
2. **Sector** objects are created for each sequence
3. Spatial data (**HeightField**, **USDZ**, **BoundaryPolygon**) is lazy-loaded on access
4. **ElevationMapGenerator** converts height fields to IsaacSim-compatible format
5. **DataConverter** handles coordinate transformations between systems
