# NaviSim Isaac Lab

Physics-based reinforcement learning for autonomous robot navigation using NVIDIA Isaac Lab.

## Overview

NaviSim Isaac Lab provides Gymnasium-compatible RL environments for training navigation agents in simulated environments using NVIDIA Isaac Sim and Isaac Lab. It supports:

- **Warehouse environment**: Static indoor environment for baseline navigation training
- **Outdoor environment**: Physics-in-simulation with your own 3D Gaussian Splatting USDZ as the visual background
- **Multi-sector streaming**: NetworkX scene graph for large environments composed of multiple USDZ sectors
- **Jetbot POV camera**: On-robot camera capture for data collection and visual navigation research

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
│   │   │   ├── __init__.py
│   │   │   ├── outdoor_env_cfg.py   #   Env config (no-camera and with-camera variants)
│   │   │   └── outdoor_scene_cfg.py #   Scene layout (ground plane + USDZ + Jetbot)
│   │   └── warehouse/
│   │       ├── warehouse_env.py     #   WarehouseEnv (DirectRLEnv)
│   │       ├── warehouse_env_cfg.py #   Environment config (obs/action spaces)
│   │       └── warehouse_scene_cfg.py #  Scene layout (assets, robot, camera)
│   ├── robots/                       # Robot configurations
│   │   └── jetbot_cfg.py            #   Jetbot ArticulationCfg
│   ├── scene/                        # Scene management
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
| `Navisim-Outdoor-Jetbot-v0` | Your USDZ (`new_point_cloud.usdz`) | No | RL training, headless-safe |
| `Navisim-Outdoor-Jetbot-Camera-v0` | Your USDZ | Yes | Camera capture; requires `--enable_cameras` |

---

## Quick Start

### 1. Integration Test

```bash
python scripts/smoke_test.py --headless
```

This launches Isaac Sim, creates the warehouse environment, runs 10 random action steps, and validates the environment lifecycle.

### 2. Train a PPO Agent

```bash
# Train with 64 parallel environments (headless)
python scripts/rsl_rl/train.py --num_envs 64 --headless

# Train with fewer envs for debugging
python scripts/rsl_rl/train.py --num_envs 4 --headless
```

Checkpoints are saved to `check_pts/rsl_rl/warehouse_jetbot/ppo/` (configurable via `--log_dir`).

Training hyperparameters are in [ppo_warehouse_jetbot.yaml](navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml).

### 3. Evaluate a Trained Agent

```bash
# Run with a trained checkpoint
python scripts/rsl_rl/play.py --checkpoint check_pts/rsl_rl/warehouse_jetbot/ppo/model_10.pt

# Run with zero policy (no checkpoint, for testing)
python scripts/rsl_rl/play.py

# Save Jetbot POV camera images during playback
python scripts/rsl_rl/play.py --checkpoint model.pt --save_camera_images --save_every 10 --output_dir outputs/playback_pov
```

### 4. Run with Camera Capture

```bash
# Outdoor environment with POV camera
python scripts/run_with_jetbot_camera.py --task Navisim-Outdoor-Jetbot-Camera-v0 --num_envs 1 --enable_cameras

# Warehouse environment with POV camera
python scripts/run_with_jetbot_camera.py --task Navisim-Warehouse-Jetbot-v0 --num_envs 1 --enable_cameras
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
| Scale | Any; Jetbot is scaled to match automatically |

> **How to check**: Open your USDZ in `usdview` or Isaac Sim, then run:
> ```python
> from pxr import UsdGeom, Usd
> stage = Usd.Stage.Open("new_point_cloud.usdz")
> print(UsdGeom.GetStageMetersPerUnit(stage))  # should be 1.0
> bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(stage.GetPseudoRoot())
> print(bb.GetRange())  # world bounding box
> ```

### Running the Outdoor Environment

```bash
# Headless RL training on your USDZ (no camera)
python scripts/rsl_rl/train.py --task Navisim-Outdoor-Jetbot-v0 --num_envs 4 --headless

# Evaluation / visualization
python scripts/rsl_rl/play.py --task Navisim-Outdoor-Jetbot-v0
```

### Jetbot Scaling

The outdoor Jetbot is configured at **scale=1.0** (real-world size, ~12 cm wide) to match a typical 3DGS reconstruction exported in SI metres. Key parameters in [outdoor_scene_cfg.py](navisim_lab/envs/outdoor/outdoor_scene_cfg.py):

```python
_OUTDOOR_JETBOT_SCALE = 1.0
_OUTDOOR_JETBOT_WHEEL_RADIUS = 0.03  # metres at scale 1.0
```

If your environment is in a different unit system (e.g., centimetres), adjust `_OUTDOOR_JETBOT_SCALE` or re-export your USDZ with `metersPerUnit=1.0`.

---

## Jetbot POV Camera

### Important: Headless vs. Camera Mode

The Jetbot camera uses Isaac Lab's `CameraCfg` sensor, which activates the RTX offscreen renderer. In headless mode **without** `--enable_cameras`, the renderer is not initialized and the simulation will **hang indefinitely** waiting for a frame.

**Rule**: Only use `Navisim-Outdoor-Jetbot-Camera-v0` (or any camera-enabled task) when you also pass `--enable_cameras` to the launcher.

| Task | `--enable_cameras` required? |
|------|-------------------------------|
| `Navisim-Outdoor-Jetbot-v0` | No (safe for headless RL training) |
| `Navisim-Outdoor-Jetbot-Camera-v0` | **Yes** |
| `Navisim-Warehouse-Jetbot-v0` | Depends on scene config used |

### Capturing Camera Images

Use the dedicated script:

```bash
python scripts/run_with_jetbot_camera.py \
    --task Navisim-Outdoor-Jetbot-Camera-v0 \
    --num_envs 1 \
    --enable_cameras
```

Images are saved to `outputs/jetbot_pov/` as PNG files named `frame_{step:06d}_env{env_id}.png`.

### Programmatic Camera Capture

```python
from navisim_lab.utils.camera_utils import get_camera_images, save_camera_batch

# After env.step():
# Get RGB images as tensor (num_envs, H, W, 3)
rgb = get_camera_images(env.unwrapped, camera_name="jetbot_camera")

# Save all environment camera images to disk
save_camera_batch(env.unwrapped, output_dir="outputs/", camera_name="jetbot_camera", step=0)
```

### Camera Architecture

The Jetbot USD includes a camera prim at `chassis/rgb_camera/jetbot_camera`. Isaac Lab requires camera prims to use canonical transform operations `[translate, orient, scale]`, but the Jetbot USD uses `rotateZYX`. This is automatically corrected by `BaseNavigationEnv._standardize_camera_transforms()` at scene setup — no manual action needed.

The camera scene config is split into two variants:

- **`OutdoorSceneCfg`** — no camera; safe for headless RL training
- **`OutdoorSceneWithCameraCfg`** — adds `jetbot_camera`; use with `--enable_cameras`

---

## Multi-Sector Navigation (NetworkX Scene Graph)

For large outdoor environments that don't fit into a single USDZ, NaviSim supports **multi-sector streaming**: the world is divided into overlapping USDZ sectors organized as a NetworkX graph. As the robot navigates, sectors are loaded and unloaded dynamically.

### Concepts

| Term | Description |
|------|-------------|
| **Sector / Section** | A single USDZ file covering a spatial region (bounding box) |
| **SceneGraph** | NetworkX graph where nodes = sectors, edges = adjacency |
| **DynamicSceneEnvWrapper** | Gymnasium wrapper that handles transitions and USD streaming |

### Building a Scene Graph

```python
from navisim_lab.scene.scene_graph import SceneGraph
import numpy as np

graph = SceneGraph()

# Add sector A (covers x: 0–5m, y: 0–7m)
graph.add_section(
    section_id="sector_A",
    usd_path="assets/sector_A.usdz",
    bounds=np.array([[0.0, 0.0, 0.0], [5.0, 7.0, 2.0]]),
    center=np.array([2.5, 3.5, 0.0]),
)

# Add sector B (adjacent, shares the x=5m boundary)
graph.add_section(
    section_id="sector_B",
    usd_path="assets/sector_B.usdz",
    bounds=np.array([[5.0, 0.0, 0.0], [10.0, 7.0, 2.0]]),
    center=np.array([7.5, 3.5, 0.0]),
    neighbors=["sector_A"],   # adds a bidirectional edge automatically
)

# Save for later reuse
graph.to_pickle("assets/scene_graph.pkl")
```

### Loading a Scene Graph from Pickle

```python
from navisim_lab.scene.scene_graph import SceneGraph
graph = SceneGraph.from_pickle("assets/scene_graph.pkl")
print(graph)  # SceneGraph(sections=2, loaded=0)
```

### Wrapping an Environment with Dynamic Streaming

```python
import gymnasium as gym
import navisim_lab.tasks  # registers task IDs

env = gym.make("Navisim-Outdoor-Jetbot-v0")
env = DynamicSceneEnvWrapper(
    env,
    scene_graph=graph,
    max_loaded_sections=3,   # keep at most 3 USDZs resident in the stage
    load_radius=20.0,        # pre-load sectors within 20m of the robot
    neighbor_depth=1,        # also load 1-hop graph neighbours
    update_frequency=10,     # run USD load/unload every 10 steps
)

obs, info = env.reset()
for _ in range(10_000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    # Detect sector transitions
    if "section_transition" in info:
        t = info["section_transition"]
        print(f"Crossed from '{t['from']}' to '{t['to']}'")
        print(f"Robot teleported to {t['teleport_target']}")
```

### Sector Transition Mechanics

When the robot crosses a sector boundary:

1. **Detection** — XY-only AABB containment check runs every step (cheap).
2. **USD swap** — New sector is loaded into the Isaac Sim stage; old sector is removed.
3. **Teleport** — Robot is moved to the new sector's center XY, preserving its current Z height and heading. Velocity is zeroed to avoid physics instability.
4. **Reward zeroing** — The reward for the transition step is set to 0 to prevent a spurious spike from the position jump.
5. **No episode reset** — The episode continues uninterrupted.

The `WarehouseEnv` reward function also has an independent teleport guard: if the robot moves more than 1m in a single step (impossible under normal physics), the reward for that step is zeroed.

### Visualizing the Scene Graph

```python
# Interactive matplotlib plot
graph.visualize()

# Save to file
graph.visualize(output_path="scene_graph.png")
```

Loaded sectors are highlighted in green; unloaded sectors in blue.

### Scene Graph Queries

```python
# Find which sector a position belongs to
section_id = graph.find_section_containing_point(np.array([3.0, 4.0, 0.1]))

# Get graph neighbors (1 hop)
neighbors = graph.get_neighbors("sector_A", depth=1)

# Find sections within a radius
nearby = graph.get_sections_within_radius(robot_pos, radius=15.0)

# Plan a path through sectors
path = graph.get_path_sections("sector_A", "sector_C")  # uses NetworkX shortest path
```

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
- Standard scene setup with `InteractiveScene`
- Camera transform standardization (see [Camera Transform Fix](#camera-transform-fix))
- Common navigation state: `prev_position`, `goal_positions`, `collision_flags`
- Utility methods: `_compute_distance_to_goal()`, `_compute_forward_progress()`
- `teleport_robot()` for sector transitions without episode reset
- Timeout-based episode termination

**WarehouseEnv** implements:
- **Observations**: 13D Jetbot root state `[pos(3), quat(4), lin_vel(3), ang_vel(3)]`
- **Actions**: 2D normalized wheel velocities `[left_wheel, right_wheel]` in `[-1, 1]`, scaled by `action_scale=5.0` rad/s
- **Rewards**: Forward progress along x-axis, weighted by `forward_reward_weight`; zeroed on teleport (|Δx| > 1m guard)
- **Reset**: Returns Jetbot to default pose with environment origin offset

### Scene Configuration

Two scene config families are provided:

**Warehouse** ([warehouse_scene_cfg.py](navisim_lab/envs/warehouse/warehouse_scene_cfg.py)):
- **warehouse**: Static warehouse USD (from Isaac Nucleus)
- **jetbot**: NVIDIA Jetbot at 7x scale for human-sized warehouse navigation
- **jetbot_camera**: POV camera (opt-in)

**Outdoor** ([outdoor_scene_cfg.py](navisim_lab/envs/outdoor/outdoor_scene_cfg.py)):
- **ground_plane**: Invisible collision surface at `/World/OutdoorGround`
- **outdoor_env**: Your USDZ visual at `/World/OutdoorEnv` (no physics geometry required)
- **jetbot**: NVIDIA Jetbot at scale=1.0 (real-world ~12 cm body width)
- **jetbot_camera**: POV camera (opt-in via `OutdoorSceneWithCameraCfg`)

### Robot Configuration

[JETBOT_CONFIG](navisim_lab/robots/jetbot_cfg.py) base parameters:
- Spawned from Isaac Nucleus Jetbot USD
- `ImplicitActuatorCfg` for `left_wheel_joint` and `right_wheel_joint`

**Warehouse scale** (7x): damping=1000, effort=5000 N·m, spawn height=0.21m

**Outdoor scale** (1x): damping=200, effort=500 N·m, spawn height=0.03m

### Task Registration

All environments are registered with Gymnasium in [tasks/warehouse/\_\_init\_\_.py](navisim_lab/tasks/warehouse/__init__.py):

```python
import navisim_lab.tasks  # Triggers registration of all task IDs
env = gym.make("Navisim-Outdoor-Jetbot-v0")
```

---

## Key Design Patterns

### AppLauncher Pattern

**Critical**: Isaac Sim must be launched before importing any Omniverse/Isaac modules.

```python
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# NOW import Isaac modules
import isaaclab.sim as sim_utils
```

### DirectRLEnv Lifecycle

```python
_setup_scene()          # Clone environments, standardize camera transforms, reset sim
_pre_physics_step()     # Process actions before physics step
_apply_action()         # Write data to simulation (scene.write_data_to_sim())
_get_observations()     # Return {"policy": obs_tensor}
_get_rewards()          # Return reward tensor
_get_dones()            # Return (terminated, truncated) tensors
_reset_idx(env_ids)     # Reset specific environments
```

> **Important**: `InteractiveScene` is created by `DirectRLEnv.__init__` *before* `_setup_scene()` is called. Do **not** create it again inside `_setup_scene()`.

### Configuration-Driven Design

- `@configclass` decorator for typed configs
- YAML for training hyperparameters (easy tuning without code changes)
- Asset paths centralized in [paths.py](navisim_lab/utils/paths.py)

---

## Camera Transform Fix

The Jetbot USD uses `rotateZYX` (Euler angles) for its camera prim transform, but Isaac Lab's `XformPrimView` requires canonical form `[translate, orient, scale]`. This causes a `ValueError` during `sim.reset()`.

**Solution**: `BaseNavigationEnv._standardize_camera_transforms()` uses the **Sdf layer API** to override the composed USD properties at the root layer level before `sim.reset()`. This converts `rotateZYX` to a quaternion (`orient`) while preserving the original transform values.

The fix runs automatically in `_setup_scene()` after `clone_environments()` and before `sim.reset()`. No manual intervention is needed.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'navisim_lab'"

Install the package:
```bash
conda activate navisim
pip install -e .
```

### "Module 'isaaclab' not found"

Ensure Isaac Sim and Isaac Lab are installed. Recreate the environment from `environment.yml` if needed.

### Simulation hangs indefinitely when camera is enabled

The camera sensor activates the RTX offscreen renderer. Without `--enable_cameras`, the renderer is absent and the Replicator annotator blocks forever waiting for a frame.

**Fix**: Always pass `--enable_cameras` when using a camera-enabled task:
```bash
python scripts/run_with_jetbot_camera.py --enable_cameras
```

Never use `Navisim-Outdoor-Jetbot-Camera-v0` without `--enable_cameras`.

### Camera transform errors ("Prim not xformable with standard transform operations")

This should be handled automatically by `BaseNavigationEnv._standardize_camera_transforms()`. If you still see this error:

1. Clear Python caches: `find . -type d -name __pycache__ -exec rm -rf {} +`
2. Verify your environment inherits from `BaseNavigationEnv`
3. Ensure `_setup_scene()` calls `self._standardize_camera_transforms()` before `self.sim.reset()`

### CUDA architecture errors ("nvrtc: error: invalid value for --gpu-architecture")

PyTorch JIT compiles CUDA kernels at runtime, and it needs to know your GPU's compute capability. If PyTorch doesn't recognize your GPU architecture, you'll see this error during `sim.reset()`.

The scripts in `scripts/rsl_rl/` already set `TORCH_CUDA_ARCH_LIST` and `PYTORCH_JIT=0` automatically for Blackwell GPUs. If you're on a different GPU or need to override:

```bash
# Find your GPU's compute capability
python -c "import torch; print(torch.cuda.get_device_capability())"
# Example output: (10, 0) for Blackwell, (8, 9) for Ada Lovelace, (8, 6) for Ampere
```

Then set `TORCH_CUDA_ARCH_LIST` to match (major.minor):

```bash
# Blackwell (RTX 50 series, GB10)
TORCH_CUDA_ARCH_LIST="10.0" PYTORCH_JIT=0 python scripts/rsl_rl/train.py --headless

# Ada Lovelace (RTX 40 series)
TORCH_CUDA_ARCH_LIST="8.9" python scripts/rsl_rl/train.py --headless

# Ampere (RTX 30 series)
TORCH_CUDA_ARCH_LIST="8.6" python scripts/rsl_rl/train.py --headless
```

To make this permanent, add to your shell profile (`~/.bashrc` or `~/.zshrc`):
```bash
export TORCH_CUDA_ARCH_LIST="10.0"  # Replace with your compute capability
export PYTORCH_JIT=0                 # Only needed if JIT still fails
```

### CUDA not available in Docker

Ensure the container is launched with `--gpus all` and X11 authorization:

```bash
docker run -it --rm \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=$XAUTHORITY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $XAUTHORITY:$XAUTHORITY \
    ...
```

### Scene assets not appearing

Check USD paths in [paths.py](navisim_lab/utils/paths.py). Verify `ISAAC_NUCLEUS_DIR` points to a valid Isaac Sim nucleus directory. For the outdoor environment, verify `assets/new_point_cloud.usdz` exists.

### Robot falls through ground

For the outdoor environment, the ground plane prim path is `/World/OutdoorGround`. Check that no prior simulation session left a conflicting prim at that path (restart Isaac Sim if needed).

For the warehouse environment, check spawn height in [jetbot_cfg.py](navisim_lab/robots/jetbot_cfg.py): should be `0.21m` for the 7x-scaled Jetbot.

### Sector transition not detected

Check that your `SceneSection.bounds` covers the area the robot is navigating. Boundary detection uses XY-only AABB (Z is ignored), so ensure the XY extents of adjacent sectors share or overlap at the boundary. Log the robot's current position and compare against `section.min_bounds` / `section.max_bounds`.

### RSL-RL step() returns 4 values, not 5

`RslRlVecEnvWrapper.step()` returns `(obs, rewards, dones, info)`, not the standard Gymnasium 5-tuple. Use:
```python
obs, rewards, dones, info = env.step(actions)
```

---

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab)
- [Isaac Lab DirectRLEnv Guide](https://isaac-sim.github.io/IsaacLab/source/tutorials/03_envs/create_direct_rl_env.html)
- [RSL-RL Repository](https://github.com/leggedrobotics/rsl_rl)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
