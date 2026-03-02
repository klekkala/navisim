# NaviSim Isaac Lab

Physics-based reinforcement learning for autonomous robot navigation using NVIDIA Isaac Lab.

## Overview

NaviSim Isaac Lab provides Gymnasium-compatible RL environments for training navigation agents in simulated environments using NVIDIA Isaac Sim and Isaac Lab. It supports:

- **Warehouse environment**: Static indoor environment for baseline navigation training
- **Outdoor environment**: Physics-in-simulation with your own 3D Gaussian Splatting USDZ as the visual background
- **Multi-sector streaming**: NetworkX scene graph for large environments composed of multiple USDZ sectors *(infrastructure ready; full integration pending)*
- **Jetbot POV camera**: On-robot camera capture for data collection and visual navigation research *(use with `--enable_cameras` only)*

## Status

### Done
- [x] Warehouse navigation task (`Navisim-Warehouse-Jetbot-v0`) — Jetbot at 7× scale, forward-progress reward, PPO training with RSL-RL
- [x] Outdoor navigation task (`Navisim-Outdoor-Jetbot`) — 3DGS USDZ as visual backdrop, invisible ground plane for physics, Jetbot at real-world scale (1×)
- [x] Jetbot POV camera capture — pass `--enable_cameras` to activate; isolated in separate config files to prevent RTX hang in headless mode
- [x] Camera/no-camera config split — camera configs live in `*_with_camera_cfg.py` files; importing the standard env never touches `isaaclab.sensors.camera`
- [x] `render_interval` = `decimation` = 2 enforced in all env configs — prevents double-render hang
- [x] `BaseNavigationEnv.teleport_robot()` — moves robot to arbitrary world position without episode reset (used by multi-sector transitions)
- [x] `SceneGraph` + `DynamicSceneManager` + `DynamicSceneEnvWrapper` — data structures and wrapper for multi-sector USD streaming
- [x] Sector transition mechanics — XY-AABB detection every step, immediate USD swap + robot teleport to sector center, reward zeroed on transition step
- [x] Teleport reward guard — `|Δx| > 1 m` in one step zeroes reward to prevent policy corruption on sector jumps

### TODO
- [ ] **End-to-end multi-sector integration test** — `DynamicSceneEnvWrapper` is implemented but not yet validated in a running Isaac Sim session; needs a two-sector smoke test
- [ ] **Reward shaping** — current reward is raw forward-x progress; needs goal-conditioned reward, heading alignment term, and obstacle penalty
- [ ] **Collision detection** — no collision termination yet; robot can phase through walls; requires contact sensor or raycasting
- [ ] **Goal conditioning** — observations do not include goal position/heading; needed for point-nav tasks
- [ ] **Multi-sector graph construction tooling** — `build_scene_graph.py` exists but no end-to-end pipeline from raw USDZ files → annotated `SceneGraph` pickle
- [ ] **Camera capture validation** — `run_with_jetbot_camera.py` has not been run successfully end-to-end since the camera/no-camera config split; verify image saving works
- [ ] **Training curriculum** — single fixed episode length; no difficulty progression or domain randomization
- [ ] **Evaluation metrics** — no structured eval loop (success rate, SPL, collision rate); only total reward is logged

---

## Prerequisites

- **NVIDIA Isaac Sim** 5.1.0
- **Isaac Lab** 2.3.2
- CUDA-capable GPU
- Python 3.11

## Installation

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate navisim
```

This installs all dependencies including Isaac Sim 5.1, Isaac Lab 2.3.2, PyTorch (CUDA 12.8), and RSL-RL.

### 2. Install the `navisim_lab` package

```bash
pip install -e .

# Verify installation
python -c "import navisim_lab; print('Success')"
```

> **Note:** Scripts in `scripts/` automatically add the project to `sys.path`, so you can also run them directly without installing the package.

---

## Folder Structure

```
navisim_isaacsim/
├── navisim_lab/                      # Main Python package
│   ├── __init__.py
│   ├── agents/                       # RL agent implementations
│   │   ├── base_agent.py            #   Abstract base class for agents
│   │   └── rsl_rl/                  #   RSL-RL specific agent
│   │       └── rsl_rl_agent.py
│   ├── camera/                       # Camera system
│   │   └── jetbot_camera.py         #   Jetbot POV camera configuration
│   ├── configs/                      # Training configurations
│   │   └── rsl_rl/
│   │       ├── ppo_cfg.py           #   Config loader for RSL-RL
│   │       └── ppo_warehouse_jetbot.yaml  # PPO hyperparameters
│   ├── data/                         # Data management
│   │   └── rocksdb_manager.py       #   RocksDB integration for spatial data
│   ├── envs/                         # RL environments
│   │   ├── base/
│   │   │   └── base_nav_env.py      #   BaseNavigationEnv (abstract base class)
│   │   ├── outdoor/                 #   Outdoor 3DGS environment
│   │   │   ├── __init__.py          #     Exports no-camera configs only (safe to import)
│   │   │   ├── outdoor_env_cfg.py   #     OutdoorEnvCfg (no camera)
│   │   │   ├── outdoor_env_with_camera_cfg.py  # OutdoorEnvWithCameraCfg (camera)
│   │   │   ├── outdoor_scene_cfg.py #     OutdoorSceneCfg (no camera)
│   │   │   └── outdoor_scene_with_camera_cfg.py # OutdoorSceneWithCameraCfg (camera)
│   │   └── warehouse/
│   │       ├── warehouse_env.py     #   WarehouseEnv (DirectRLEnv)
│   │       ├── warehouse_env_cfg.py #   Environment config (obs/action spaces)
│   │       └── warehouse_scene_cfg.py #  Scene layout (no camera)
│   ├── robots/                       # Robot configurations
│   │   └── jetbot_cfg.py            #   Jetbot ArticulationCfg
│   ├── scene/                        # Scene management (multi-sector, not yet integrated)
│   │   ├── build_scene_graph.py     #   Scene graph construction helpers
│   │   ├── dynamic_scene_env_wrapper.py # Wrapper for multi-sector streaming
│   │   ├── dynamic_scene_manager.py #   Runtime USD load/unload
│   │   ├── scene_graph.py           #   SceneGraph + SceneSection data structures
│   │   └── sequence_graph.py        #   Sequence graph integration
│   ├── tasks/                        # Gymnasium task registry
│   │   ├── __init__.py              #   Imports warehouse tasks
│   │   └── warehouse/
│   │       └── __init__.py          #   Registers all Gymnasium task IDs
│   └── utils/                        # Utilities
│       ├── camera_utils.py          #   Camera capture and image saving
│       ├── paths.py                 #   Isaac Nucleus + custom asset paths
│       └── sequence_graph_tools.py  #   Sequence graph navigation tools
├── assets/                           # Custom USD/USDZ assets (not tracked in git)
│   └── new_point_cloud.usdz         #   Your 3DGS-converted outdoor scene
├── scripts/                          # Executable scripts
│   ├── smoke_test.py                #   Quick integration test
│   ├── run_with_jetbot_camera.py    #   Run env with camera capture
│   └── rsl_rl/                      #   RSL-RL training/evaluation
│       ├── train.py                 #     Train PPO agent
│       └── play.py                  #     Evaluate trained agent
├── tests/                            # Test suite
├── examples/                         # Example scripts
├── environment.yml                   # Conda environment specification
├── pyproject.toml                    # Package metadata
└── README.md                         # This file
```

---

## Registered Gymnasium Task IDs

| Task ID | Environment | Camera | Notes |
|---------|-------------|--------|-------|
| `Navisim-Warehouse-Jetbot-v0` | Static warehouse USD | No | Baseline indoor navigation |
| `Navisim-Outdoor-Jetbot` | Your USDZ (`new_point_cloud.usdz`) | Optional | RL training; add `--enable_cameras` to activate Jetbot POV camera |

---

## Quick Start

### 1. Integration Test

```bash
python scripts/smoke_test.py --headless
```

This launches Isaac Sim, creates the warehouse environment, runs 10 random action steps, and validates the environment lifecycle.

### 2. Train a PPO Agent

```bash
# Warehouse (default task)
python scripts/rsl_rl/train.py --num_envs 64 --headless

# Outdoor 3DGS environment
python scripts/rsl_rl/train.py --task Navisim-Outdoor-Jetbot --num_envs 4 --headless
```

Checkpoints are saved to `check_pts/rsl_rl/warehouse_jetbot/ppo/` (configurable via `--log_dir`).

Training hyperparameters are in [ppo_warehouse_jetbot.yaml](navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml).

### 3. Evaluate a Trained Agent

```bash
# Run with a trained checkpoint
python scripts/rsl_rl/play.py --checkpoint check_pts/rsl_rl/warehouse_jetbot/ppo/model_10.pt

# Run with zero policy (no checkpoint, for testing)
python scripts/rsl_rl/play.py

# Outdoor task
python scripts/rsl_rl/play.py --task Navisim-Outdoor-Jetbot
```

### 4. Run with Camera Capture

```bash
# Outdoor task without camera (headless-safe)
python scripts/run_with_jetbot_camera.py --task Navisim-Outdoor-Jetbot

# Outdoor task WITH Jetbot POV camera capture
python scripts/run_with_jetbot_camera.py --task Navisim-Outdoor-Jetbot --enable_cameras
```

Camera frames are saved to `outputs/jetbot_pov/`.

---

## Using Your Own USDZ (Outdoor Environment)

### Overview

The outdoor environment loads a 3D Gaussian Splatting (3DGS) USDZ file as a **purely visual** background. Because USDZ converted from 3DGS contains no collision geometry or rigid bodies, physics is handled separately:

- An invisible **ground plane** (`GroundPlaneCfg`) provides the collision surface for the Jetbot wheels.
- The **USDZ** is loaded as a static visual reference only.
- `SimulationContext` (created automatically by `DirectRLEnv`) provides the physics scene.

### Downloading the Sample USDZ

A sample 3DGS-converted outdoor scene is available for download:

**[Download new_point_cloud.usdz (Google Drive)](https://drive.google.com/file/d/1-0rLtj1qhKbFaMejgJJsn097Aucyrtpt/view?usp=drive_link)**

After downloading, place it in the `assets/` directory inside `navisim_isaacsim/`:

```
navisim_isaacsim/
└── assets/
    └── new_point_cloud.usdz   ← place it here
```

```bash
# From the repo root
mkdir -p navisim_isaacsim/assets
mv ~/Downloads/new_point_cloud.usdz navisim_isaacsim/assets/new_point_cloud.usdz
```

### Placing Your Own USDZ

To use your own scene instead of the sample, copy it to the same location:

```bash
cp /path/to/your/scene.usdz navisim_isaacsim/assets/new_point_cloud.usdz
```

The path is configured in [paths.py](navisim_lab/utils/paths.py):

```python
CUSTOM_ENV_USD = str(Path(__file__).parents[2] / "assets" / "new_point_cloud.usdz")
```

To use a different filename or path, edit this constant.

### USDZ Requirements

| Property | Requirement |
|----------|-------------|
| Coordinate system | Z-up |
| Meters per unit | 1.0 (SI units; 1 unit = 1 metre) |
| Physics geometry | Not required (ground plane provides collision) |
| Scale | Any; adjust `_OUTDOOR_JETBOT_SCALE` in `outdoor_scene_cfg.py` to match |

> **How to check**: Open your USDZ in `usdview` or Isaac Sim, then run:
> ```python
> from pxr import UsdGeom, Usd
> stage = Usd.Stage.Open("new_point_cloud.usdz")
> print(UsdGeom.GetStageMetersPerUnit(stage))  # should be 1.0
> bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(stage.GetPseudoRoot())
> print(bb.GetRange())  # world bounding box
> ```

### Jetbot Scaling

The outdoor Jetbot is configured at **scale=1.0** (real-world size, ~12 cm wide) to match a typical 3DGS reconstruction exported in SI metres. Key parameters in [outdoor_scene_cfg.py](navisim_lab/envs/outdoor/outdoor_scene_cfg.py):

```python
_OUTDOOR_JETBOT_SCALE = 1.0
_OUTDOOR_JETBOT_WHEEL_RADIUS = 0.03  # metres at scale 1.0
```

If your environment is in a different unit system (e.g., centimetres), adjust `_OUTDOOR_JETBOT_SCALE` or re-export your USDZ with `metersPerUnit=1.0`.

---

## Jetbot POV Camera

### Important: Camera Configs Are Isolated

`CameraCfg` from Isaac Lab sets the carb flag `rtx_sensors=True` the moment it is instantiated — even if the camera is never used. This activates the RTX offscreen renderer, which causes the simulation to **hang indefinitely** in headless mode without `--enable_cameras`.

To prevent this, camera configs are kept in **separate files** that are never imported by the standard training path:

| File | Imports camera? | When to use |
|------|----------------|-------------|
| `outdoor_scene_cfg.py` | No | All RL training / eval |
| `outdoor_scene_with_camera_cfg.py` | Yes | Camera capture only |
| `outdoor_env_cfg.py` | No | All RL training / eval |
| `outdoor_env_with_camera_cfg.py` | Yes | Camera capture only |

**Rule**: Only pass `--enable_cameras` when you want the Jetbot POV camera active.

| Task | `--enable_cameras` needed? |
|------|----------------------------|
| `Navisim-Warehouse-Jetbot-v0` | No |
| `Navisim-Outdoor-Jetbot` (no camera) | No |
| `Navisim-Outdoor-Jetbot` + `--enable_cameras` | **Yes** |

### Capturing Camera Images

```bash
python scripts/run_with_jetbot_camera.py --enable_cameras
```

Images are saved to `outputs/jetbot_pov/` as PNG files.

### Programmatic Camera Capture

```python
# Must be used with a camera-enabled env config and --enable_cameras
from navisim_lab.utils.camera_utils import get_camera_images, save_camera_batch

rgb = get_camera_images(env.unwrapped, camera_name="jetbot_camera")
save_camera_batch(env.unwrapped, output_dir="outputs/", camera_name="jetbot_camera", step=0)
```

### Camera Transform Fix

The Jetbot USD uses `rotateZYX` (Euler angles) for its camera prim transform, but Isaac Lab requires `[translate, orient, scale]`. `BaseNavigationEnv._standardize_camera_transforms()` corrects this at the Sdf layer level before `sim.reset()`. It runs automatically and only when a `jetbot_camera` attribute is present in the scene config.

---

## Multi-Sector Navigation (NetworkX Scene Graph)

> **Status**: Infrastructure is implemented and tested in isolation. End-to-end integration with a running Isaac Sim session is pending. Do not use `DynamicSceneEnvWrapper` in production yet.

The data structures and wrapper for streaming large environments as a graph of USDZ sectors are ready in `navisim_lab/scene/`. The design is documented here for reference.

### Concepts

| Term | Description |
|------|-------------|
| **Sector / Section** | A single USDZ file covering a spatial bounding box |
| **SceneGraph** | NetworkX graph: nodes = sectors, edges = spatial adjacency |
| **DynamicSceneEnvWrapper** | Gymnasium wrapper for transition detection and USD streaming |

### Building a Scene Graph

```python
from navisim_lab.scene.scene_graph import SceneGraph
import numpy as np

graph = SceneGraph()

graph.add_section(
    section_id="sector_A",
    usd_path="assets/sector_A.usdz",
    bounds=np.array([[0.0, 0.0, 0.0], [5.0, 7.0, 2.0]]),
    center=np.array([2.5, 3.5, 0.0]),
)
graph.add_section(
    section_id="sector_B",
    usd_path="assets/sector_B.usdz",
    bounds=np.array([[5.0, 0.0, 0.0], [10.0, 7.0, 2.0]]),
    center=np.array([7.5, 3.5, 0.0]),
    neighbors=["sector_A"],
)
graph.to_pickle("assets/scene_graph.pkl")
```

### Wrapping an Environment

```python
from navisim_lab.scene import DynamicSceneEnvWrapper

env = gym.make("Navisim-Outdoor-Jetbot")
env = DynamicSceneEnvWrapper(
    env,
    scene_graph=graph,
    max_loaded_sections=3,
    load_radius=20.0,
    update_frequency=10,
)
```

### Sector Transition Mechanics

When the robot crosses a sector boundary:

1. **Detection** — XY-only AABB containment check runs every step.
2. **USD swap** — New sector loaded; old sector removed from stage.
3. **Teleport** — Robot moved to new sector's XY center, heading preserved, velocity zeroed.
4. **Reward zeroed** — Prevents reward spike from the position jump.
5. **No episode reset** — Episode continues uninterrupted.

---

## Architecture

### Environment Hierarchy

```
DirectRLEnv (Isaac Lab)
  └── BaseNavigationEnv (base_nav_env.py)
        └── WarehouseEnv (warehouse_env.py)
              Used by both Warehouse and Outdoor task configs
```

**BaseNavigationEnv** provides:
- Scene setup with `clone_environments()` → camera transform fix → `sim.reset()`
- Camera transform standardization (`rotateZYX` → `orient`) for camera-enabled configs
- Common navigation state: `prev_position`, `goal_positions`, `collision_flags`
- `teleport_robot()` for sector transitions without episode reset
- Timeout-based episode termination via `_get_dones()`

**WarehouseEnv** implements:
- **Observations**: 13D Jetbot root state `[pos(3), quat(4), lin_vel(3), ang_vel(3)]`
- **Actions**: 2D normalized wheel velocities `[left_wheel, right_wheel]` in `[-1, 1]`, scaled by `action_scale=5.0` rad/s
- **Rewards**: Forward progress along x-axis; zeroed when `|Δx| > 1 m` (teleport guard)
- **Reset**: Returns Jetbot to default pose + environment origin offset

### Scene Configuration

**Warehouse** ([warehouse_scene_cfg.py](navisim_lab/envs/warehouse/warehouse_scene_cfg.py)):
- `warehouse`: Static warehouse USD from Isaac Nucleus
- `jetbot`: Jetbot at 7× scale (human-sized indoor navigation)

**Outdoor** ([outdoor_scene_cfg.py](navisim_lab/envs/outdoor/outdoor_scene_cfg.py)):
- `ground_plane`: Collision surface at `/World/OutdoorGround`
- `outdoor_env`: USDZ visual reference at `/World/OutdoorEnv`
- `jetbot`: Jetbot at 1× scale (~12 cm body width, real-world SI units)

### Robot Parameters

| Scale | Damping | Effort limit | Spawn height |
|-------|---------|--------------|--------------|
| 7× (warehouse) | 1000 | 5000 N·m | 0.21 m |
| 1× (outdoor) | 200 | 500 N·m | 0.03 m |

### Key Implementation Notes

- `render_interval` must equal `decimation` (both = 2) — one `sim.render()` call per env step. A mismatch causes a second render call that blocks on a missing Replicator frame.
- `InteractiveScene` is created by `DirectRLEnv.__init__` *before* `_setup_scene()`. Never create it again inside `_setup_scene()`.
- Camera module imports must never appear at the top level of files that are imported during normal (no-camera) startup — they set `rtx_sensors=True` globally.

---

## Troubleshooting

### Simulation hangs indefinitely

Most likely cause: the RTX offscreen renderer was activated (by a `CameraCfg` import or `args.enable_cameras = True`) but `--enable_cameras` was not passed to `AppLauncher`.

Check:
1. Are you importing `outdoor_scene_with_camera_cfg` or `outdoor_env_with_camera_cfg` anywhere in the startup path? These files must only be imported when using a camera task.
2. Is any script setting `args.enable_cameras = True` before `AppLauncher`? The train/play scripts no longer do this — pass `--enable_cameras` explicitly on the command line if needed.
3. Does `render_interval` equal `decimation` in your env config? Both should be `2`.

### "ModuleNotFoundError: No module named 'navisim_lab'"

```bash
conda activate navisim
pip install -e .
```

### "Module 'isaaclab' not found"

Ensure Isaac Sim and Isaac Lab are installed. Recreate the environment from `environment.yml` if needed.

### Camera transform errors ("Prim not xformable with standard transform operations")

Handled automatically by `BaseNavigationEnv._standardize_camera_transforms()`. If still occurring:
1. Clear Python caches: `find . -type d -name __pycache__ -exec rm -rf {} +`
2. Verify the env inherits from `BaseNavigationEnv`
3. Verify `_setup_scene()` calls `_standardize_camera_transforms()` before `sim.reset()`

### CUDA architecture errors ("nvrtc: error: invalid value for --gpu-architecture")

```bash
# Find your GPU's compute capability
python -c "import torch; print(torch.cuda.get_device_capability())"
```

Set `TORCH_CUDA_ARCH_LIST` to match (e.g., `10.0` for Blackwell, `8.9` for Ada Lovelace):

```bash
TORCH_CUDA_ARCH_LIST="10.0" PYTORCH_JIT=0 python scripts/rsl_rl/train.py --headless
```

To make permanent, add to `~/.bashrc`:
```bash
export TORCH_CUDA_ARCH_LIST="10.0"
export PYTORCH_JIT=0
```

### Robot falls through ground

Outdoor: verify `assets/new_point_cloud.usdz` is present and `metersPerUnit=1.0`. The ground plane is at `/World/OutdoorGround`.

Warehouse: spawn height should be `0.21 m` (wheel radius 0.03 m × 7× scale).

### RSL-RL step() returns 4 values, not 5

`RslRlVecEnvWrapper.step()` returns `(obs, rewards, dones, info)`:
```python
obs, rewards, dones, info = env.step(actions)
```

---

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab)
- [Isaac Lab DirectRLEnv Guide](https://isaac-sim.github.io/IsaacLab/source/tutorials/03_envs/create_direct_rl_env.html)
- [RSL-RL Repository](https://github.com/leggedrobotics/rsl_rl)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
