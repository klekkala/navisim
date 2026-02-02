# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Navisim is a high-fidelity simulation toolkit for evaluating autonomous navigation agents in realistic outdoor environments. It combines real-world data from mobile robots with GPU-accelerated Gaussian Splatting rendering and provides a Gymnasium-compatible interface for RL workflows.

The repository contains two main packages:
1. **navisim**: Core simulation toolkit with rendering, spatial data structures, and environment management
2. **navisim_isaacsim**: Integration layer for Isaac Sim for physics-based RL navigation

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

### Isaac Integration (navisim_isaacsim/)

**Overview**: Integration layer built on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab) for physics-based RL navigation with photorealistic rendering from NaviSim.

**Current Implementation**:
- `navisim_lab/envs/warehouse/warehouse_env.py`: WarehouseEnv inheriting from DirectRLEnv for Jetbot navigation
- `navisim_lab/envs/warehouse/warehouse_env_cfg.py`: WarehouseEnvCfg with environment configuration (13D observations, 2D actions)
- `navisim_lab/envs/warehouse/warehouse_scene_cfg.py`: WarehouseSceneCfg defining scene layout (warehouse, lighting, robot)
- `navisim_lab/robots/jetbot_cfg.py`: ArticulationCfg for Jetbot robot with ImplicitActuatorCfg for wheel control
- `navisim_lab/camera/jetbot_camera.py`: POV camera implementation for Jetbot
- `navisim_lab/utils/paths.py`: Isaac Nucleus asset paths (warehouse USD, Jetbot USD)
- Scripts (if present in parent `scripts/` directory): AppLauncher-based scripts to run tasks

**IsaacLab Architecture Patterns**:

*Scene Configuration*: Uses `InteractiveSceneCfg` with declarative asset definitions
- Assets defined as class attributes (e.g., `warehouse`, `jetbot`, `dome_light`)
- Spawned via `sim_utils.UsdFileCfg` for USD assets or procedural spawn configs
- Articulations use `ArticulationCfg` with actuator definitions via regex joint patterns

*DirectRLEnv Implementation*: Inherits from `DirectRLEnv` for optimized RL workflow
- `_setup_scene()`: Creates InteractiveScene, clones environments, resets simulation
- `_pre_physics_step(actions)`: Scales and applies actions to actuators before physics step
- `_apply_action()`: Writes data to simulation (calls `scene.write_data_to_sim()`)
- `_get_observations()`: Returns dict with 'policy' key containing observation tensor (13D: position, quaternion, velocities)
- `_get_rewards()`: Computes reward tensor based on forward progress
- Uses PyTorch tensors for all state/action/reward data on GPU

*Running Tasks*:
```bash
cd navisim_isaacsim
# Example command (adjust based on actual scripts present)
python scripts/run_with_jetbot_camera.py --num_envs 1
python scripts/smoke_test.py
```
AppLauncher handles Isaac Sim initialization and provides simulation_app instance

**IsaacLab Key Concepts**:

*Manager-Based vs Direct Workflow*:
- **Manager-Based**: Modular approach using ObservationManager, ActionManager, RewardManager, EventManager
  - Promotes collaboration and configuration-driven development
  - Environment inherits from `ManagerBasedRLEnv`
- **Direct Workflow**: All logic in single task class, more transparent, better for optimization
  - Inherits from `DirectRLEnv`, implements `_get_observations()`, `_get_rewards()`, `_apply_action()`, etc.
  - **Current implementation uses Direct Workflow** (WarehouseEnv inherits from DirectRLEnv)

*InteractiveScene*: Central scene management system
- Handles multi-environment cloning and spacing via `env_spacing` parameter
- Provides dictionary-style access to assets (e.g., `scene["jetbot"]`)
- Manages `env_origins` for parallelized environment offsets
- Call `scene.update(dt)` after each physics step to refresh data

*Articulation API*:
- `write_root_pose_to_sim()` / `write_joint_state_to_sim()`: Write states to simulation
- `set_joint_velocity_target()`: Apply velocity commands (for ImplicitActuator)
- `data.root_state_w`: World-frame root state tensor (pos, quat, lin_vel, ang_vel)
- `data.default_root_state` / `data.default_joint_pos`: Reset defaults

*Environment Registration*:
- Use `gymnasium.register()` with `"Isaac-TaskName-RobotName-v0"` naming convention
- Specify `entry_point` (env class) and `env_cfg_entry_point` (config class or YAML)
- Import package `__init__.py` to trigger registration before `gymnasium.make()`

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

### Working with IsaacLab Tasks

**Running the warehouse navigation task**:
```bash
cd navisim_isaacsim
# Run available scripts (check ../scripts/ directory for current options)
python ../scripts/smoke_test.py
python ../scripts/run_with_jetbot_camera.py --num_envs 1 --headless

# RL training with RSL-RL (if scripts exist)
python ../scripts/rsl_rl/train.py --num_envs 4
python ../scripts/rsl_rl/play.py --num_envs 1
```

**RL Training Configuration**:
The project uses RSL-RL for reinforcement learning training:
- Config: `navisim_lab/configs/rsl_rl/ppo_cfg.py` and `ppo_warehouse_jetbot.yaml`
- The config uses `RslRlOnPolicyRunnerCfg` with nested `RslRlPpoActorCriticCfg` and `RslRlPpoAlgorithmCfg`
- YAML-based configuration loaded via `isaaclab.utils.io.load_yaml`
- Training scripts integrate with Isaac Lab's RL workflow

**Adding new assets to scene**:
1. Define asset config in `navisim_lab/robots/` (e.g., new robot ArticulationCfg)
2. Add to scene config in `navisim_lab/envs/warehouse/warehouse_scene_cfg.py` as class attribute
3. Access in environment via `self.scene["asset_name"]`

**Modifying task behavior**:
1. Edit `navisim_lab/envs/warehouse/warehouse_env.py`
2. Modify `_get_rewards()` for custom reward logic
3. Update `_get_observations()` to change observation structure
4. Adjust `_pre_physics_step()` for different action processing

**Creating new scenes**:
1. Inherit from `InteractiveSceneCfg` in `navisim_lab/envs/`
2. Define assets as class attributes with spawn configs
3. Use `sim_utils.UsdFileCfg`, `GroundPlaneCfg`, or other spawn utilities
4. Instantiate in environment's `_setup_scene()`: `InteractiveScene(self.cfg.scene)`

**Working with cameras**:
The project includes a modular camera system for POV rendering:
- `navisim_lab/camera/jetbot_camera.py`: Jetbot POV camera implementation
- Camera is attached to robot and can capture frames during simulation
- Use `camera.capture()` to get RGB frames from the robot's perspective
- Camera module is separate from the Isaac Lab Camera sensor for flexibility

### Git Workflow

Branch naming convention:
- `feat/*` — new features
- `fix/*` — bug fixes
- `chore/*` — refactoring, configuration, cleanup
- `hotfix/*` — urgent production patches

Main branch: `main`

## Important Notes

**Core NaviSim**:
- **CUDA Support**: Optional but recommended for rendering performance. Falls back to CPU if unavailable.
- **Gaussian Splatting Dependency**: Requires `third_party/gaussian_splatting` cloned via `make submodule-init`
- **Asset Requirements**: Download `sequence_graph.gpickle` and `database.tar` from Google Drive (see README) before running environments
- **Memory Management**: Always call `sector.unload_all()` when switching sectors to prevent memory leaks
- **RocksDB Reset**: Call `reset_db()` in environment reset to ensure clean database state
- **Python Version**: Requires Python 3.10 (specified in environment.yaml)

**IsaacLab Integration**:
- **Isaac Sim Required**: IsaacLab requires NVIDIA Isaac Sim installation (separate from this repo)
- **AppLauncher Pattern**: Always use `AppLauncher` at script start before importing Isaac modules
- **Tensor Device**: All state/action tensors are on GPU by default (`device="cuda:0"`), use `.cpu()` for numpy conversion
- **Scene Update**: Must call `scene.update(dt)` after every `sim.step()` to refresh asset data
- **Environment Origins**: Multi-env scenarios use `scene.env_origins` to offset positions, add this when resetting root states
- **Isaac Nucleus Assets**: Built-in assets accessible via `ISAAC_NUCLEUS_DIR` path from `isaaclab.utils.assets`

## Troubleshooting

**Core NaviSim**:

**C++ compilation errors**: Ensure pybind11 is installed and C++14 compatible compiler is available

**Import errors for gaussian_splatting**: Run `make submodule-init` to clone the dependency

**RocksDB path issues**: Database auto-discovery looks for setup.py in parent directories. Ensure you're running from project root or a subdirectory.

**Memory issues during rendering**: Check that sectors are being unloaded properly. Use `sector.unload_all()` to free GPU memory.

**CUDA not available warnings**: Optional. Package works in CPU-only mode but with reduced rendering performance.

**IsaacLab Integration**:

**"Module 'isaaclab' not found"**: Ensure Isaac Sim and Isaac Lab are properly installed. Check PYTHONPATH includes Isaac Lab location.

**AppLauncher import errors**: Import AppLauncher before any Isaac/Omniverse modules. Move `from isaaclab.app import AppLauncher` to top of script.

**Scene assets not appearing**: Check USD paths in `configs/paths.py`. Verify `ISAAC_NUCLEUS_DIR` points to valid Isaac Sim nucleus directory.

**PhysX warnings**: Normal for initial setup. Ensure `sim.reset()` is called after creating InteractiveScene.

**Headless mode issues**: Use `--headless` flag with AppLauncher. Some features (like cameras) may require viewer mode.
