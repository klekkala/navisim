# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Navisim is a high-fidelity simulation toolkit for evaluating autonomous navigation agents in realistic outdoor environments. It combines real-world data from mobile robots with GPU-accelerated Gaussian Splatting rendering and provides a Gymnasium-compatible interface for RL workflows.

The repository contains two main packages:
1. **navisim**: Core simulation toolkit with rendering, spatial data structures, and environment management
2. **navisim_isaac**: Integration layer for Isaac Sim for physics-based RL navigation

## Development Environment

### Initial Setup

```bash
# Create conda environment
conda env update -f environment.yaml
conda activate navisim

# Complete setup (dependencies + submodules + C++ bindings)
make setup

# Or step by step:
make submodule-init  # Clone gaussian-splatting dependency
make install         # Install conda dependencies
make cpp_binding     # Build C++ extensions
```

### Building C++ Extensions

The project includes C++ extensions in `native/` for performance-critical operations (pose transformations). After modifying C++ code:

```bash
# Rebuild in-place
python setup.py build_ext --inplace

# Or use make
make cpp_binding

# Test the module
python test_cpp_module.py
```

### Running Tests

```bash
# Run specific test files
python tests/test_elevation_map.py
python tests/test_navisim_arena.py

# Test rendering (requires gaussian-splatting submodule)
python test_render.py

# Test C++ bindings
python test_cpp_module.py
```

### Cleaning Build Artifacts

```bash
make clean     # Remove all build artifacts and .so files
make rebuild   # Clean + rebuild C++ bindings
```

## Architecture

### Core Components (navisim/)

**World Representation**:
- `world/sequence_graph.py`: NetworkX-based navigation graph loaded from pickle, manages sequences composed of sectors
- `world/sector.py`: Represents a navigation sector with lazy-loaded spatial data (elevation map, occupancy map, boundary polygon, Gaussian model)

**Rendering Pipeline**:
- `render/navisim_scene.py`: Scene management for rendering
- `render/navisim_camera.py`: Camera projection and pose management
- `render/gaussian_splatting.py`: Integration with gaussian-splatting for photorealistic rendering
- Uses `third_party/gaussian_splatting` submodule (graphdeco-inria/gaussian-splatting)

**Spatial Data Structures**:
- `spaces/elevation_map.py`: Height field data for terrain
- `spaces/occupancy_map.py`: Occupancy grid for navigation
- `spaces/boundary_polygon.py`: Spatial boundary constraints
- All spatial data is lazy-loaded per sector to manage memory

**Data Management**:
- `data/rocksdb.py`: RocksDB singleton for storing/retrieving Gaussian models and spatial data
- Database path: `assets/rocksdb` (auto-detected from project root)
- Required assets: `assets/sequence_graph.gpickle`, `assets/database.tar` (extract before use)

**Environment Interface**:
- `envs/navisim_env.py`: Main Gymnasium-compatible environment
- `motion/simple_motion_model.py`: Trajectory simulation and motion models
- `agents/`: Agent controllers and pipelines

**Configuration**:
- `config/gaussian_model_param.py`: Gaussian Splatting parameters
- `config/pipeline.py`: Pipeline configuration

### Isaac Integration (navisim_isaac/)

**NaviSim Components** (`navisim_isaac/navisim/`):
- `world/sequence_graph.py`: Loads navigation graph from pickle
- `world/sector.py`: Sector with lazy-loaded height field, USDZ, and boundary data
- `spaces/`: Height fields, USDZ loader, boundary polygons, elevation map generator
- `converters.py`: Data format converters between NaviSim and IsaacSim

**IsaacSim Integration** (`navisim_isaac/isaac/`):
- `simulator.py`: Main IsaacSim environment interface
- `physx.py`: PhysX collision detection
- `terrain.py`: Terrain and height field management

**RL Interface** (`navisim_isaac/gymnasium/`):
- `env.py`: Gymnasium-compatible environment
- `agent.py`: Navigation agent interface
- `rl.py`: RL training utilities

**Configuration**:
- `config/default_config.yaml`: YAML configuration for all components

## Key Design Patterns

### Lazy Loading for Memory Management

Sectors use Python properties to lazy-load spatial data only when accessed:

```python
# Sector loads data on first access
sector.elevation_map  # Loads from RocksDB if not already loaded
sector.gaussian_model  # Loads Gaussian Splatting model
sector.unload_all()   # Free memory when switching sectors
```

This pattern is critical for handling large-scale environments with multiple sectors.

### Database Singleton Pattern

RocksDB is accessed through a singleton pattern in `data/rocksdb.py`:
- Thread-safe initialization
- Auto-discovery of project root
- Lazy connection management
- `reset_db()` function to close/reopen connection (used in env.reset())

### Sequence Graph Navigation

The sequence graph represents navigation routes as a NetworkX graph:
- Nodes = sequence IDs (e.g., "2024-01-15/session_01")
- Each sequence contains an ordered list of sectors
- Sectors maintain prev/next pointers for traversal
- Graph loaded from `assets/sequence_graph.gpickle`

## Common Development Workflows

### Testing Rendering Performance

```bash
# Benchmark Gaussian Splatting rendering
python test_render.py
# Expected: ~110 FPS at 1280x720 resolution (single-threaded)
```

### Working with Spatial Data

When adding new spatial data types:
1. Create a new class in `navisim/spaces/`
2. Add lazy-loading property to `world/sector.py`
3. Implement RocksDB serialization in `data/rocksdb.py`
4. Add to `sector.unload_all()` for memory management

### Modifying C++ Extensions

1. Edit files in `native/` (e.g., `binding.cpp`)
2. Rebuild: `make cpp_binding`
3. Test: `python test_cpp_module.py`
4. Important: Include paths are configured in `setup.py` via `Pybind11Extension`

### Git Workflow

Branch naming convention:
- `feat/*` — new features
- `fix/*` — bug fixes
- `chore/*` — refactoring, configuration, cleanup
- `hotfix/*` — urgent production patches

Main branch: `main`

## Important Notes

- **CUDA Support**: Optional but recommended for rendering performance. Falls back to CPU if unavailable.
- **Gaussian Splatting Dependency**: Requires `third_party/gaussian_splatting` cloned via `make submodule-init`
- **Asset Requirements**: Download `sequence_graph.gpickle` and `database.tar` from Google Drive (see README) before running environments
- **Memory Management**: Always call `sector.unload_all()` when switching sectors to prevent memory leaks
- **RocksDB Reset**: Call `reset_db()` in environment reset to ensure clean database state
- **Python Version**: Requires Python 3.10 (specified in environment.yaml)

## Troubleshooting

**C++ compilation errors**: Ensure pybind11 is installed and C++14 compatible compiler is available

**Import errors for gaussian_splatting**: Run `make submodule-init` to clone the dependency

**RocksDB path issues**: Database auto-discovery looks for setup.py in parent directories. Ensure you're running from project root or a subdirectory.

**Memory issues during rendering**: Check that sectors are being unloaded properly. Use `sector.unload_all()` to free GPU memory.

**CUDA not available warnings**: Optional. Package works in CPU-only mode but with reduced rendering performance.
