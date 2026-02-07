# NaviSim Isaac Lab

Physics-based reinforcement learning for autonomous robot navigation using NVIDIA Isaac Lab.

## Overview

NaviSim Isaac Lab provides Gymnasium-compatible RL environments for training navigation agents in simulated environments using NVIDIA Isaac Sim and Isaac Lab. It currently implements a warehouse navigation task with the Jetbot robot platform.

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
│   │   └── warehouse/
│   │       ├── warehouse_env.py     #   WarehouseEnv (DirectRLEnv)
│   │       ├── warehouse_env_cfg.py #   Environment config (obs/action spaces)
│   │       └── warehouse_scene_cfg.py #  Scene layout (assets, robot, camera)
│   ├── robots/                       # Robot configurations
│   │   └── jetbot_cfg.py            #   Jetbot ArticulationCfg
│   ├── scene/                        # Scene management
│   │   ├── build_scene_graph.py     #   Scene graph construction
│   │   ├── dynamic_scene_manager.py #   Runtime scene management
│   │   ├── scene_graph.py           #   Scene graph data structure
│   │   └── sequence_graph.py        #   Sequence graph integration
│   ├── tasks/                        # Gymnasium task registry
│   │   ├── __init__.py              #   Imports warehouse tasks
│   │   └── warehouse/
│   │       └── __init__.py          #   Registers "Navisim-Warehouse-Jetbot-v0"
│   └── utils/                        # Utilities
│       ├── camera_utils.py          #   Camera capture and image saving
│       ├── paths.py                 #   Isaac Nucleus asset paths
│       └── sequence_graph_tools.py  #   Sequence graph navigation tools
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
python scripts/run_with_jetbot_camera.py --num_envs 1
```

Camera frames are saved to `outputs/jetbot_pov/`.

---

## Architecture

### Environment Hierarchy

```
DirectRLEnv (Isaac Lab)
  └── BaseNavigationEnv (base_nav_env.py)
        └── WarehouseEnv (warehouse_env.py)
```

**BaseNavigationEnv** provides:
- Standard scene setup with `InteractiveScene`
- Camera transform standardization (see [Camera Transform Fix](#camera-transform-fix))
- Common navigation state: `prev_position`, `goal_positions`, `collision_flags`
- Utility methods: `_compute_distance_to_goal()`, `_compute_forward_progress()`
- Timeout-based episode termination

**WarehouseEnv** implements:
- **Observations**: 13D Jetbot root state `[pos(3), quat(4), lin_vel(3), ang_vel(3)]`
- **Actions**: 2D normalized wheel velocities `[left_wheel, right_wheel]` in `[-1, 1]`, scaled by `action_scale=5.0` rad/s
- **Rewards**: Forward progress along x-axis, weighted by `forward_reward_weight`
- **Reset**: Returns Jetbot to default pose with environment origin offset

### Scene Configuration

[WarehouseSceneCfg](navisim_lab/envs/warehouse/warehouse_scene_cfg.py) defines:
- **warehouse**: Static warehouse USD environment
- **jetbot**: NVIDIA Jetbot robot (7x scale for human-sized navigation)
- **jetbot_camera**: POV camera referencing the existing camera prim in the Jetbot USD

### Robot Configuration

[JETBOT_CONFIG](navisim_lab/robots/jetbot_cfg.py):
- Spawned from Isaac Nucleus Jetbot USD at 7x scale
- `ImplicitActuatorCfg` for `left_wheel_joint` and `right_wheel_joint`
- High damping (1000.0) and effort limit (5000 N·m) to handle scaled mass
- Spawn height: 0.21m (wheel radius 0.03m x 7)

### Task Registration

The environment is registered with Gymnasium as `Navisim-Warehouse-Jetbot-v0` in [tasks/warehouse/\_\_init\_\_.py](navisim_lab/tasks/warehouse/__init__.py):

```python
import navisim_lab.tasks  # Triggers registration
env = gym.make("Navisim-Warehouse-Jetbot-v0", cfg=env_cfg)
```

### Camera System

The Jetbot USD includes a camera at `chassis/rgb_camera/jetbot_camera`. To use it:

1. **Configuration**: [jetbot_camera.py](navisim_lab/camera/jetbot_camera.py) defines a `CameraCfg` with `spawn=None` (references existing prim) and `update_latest_camera_pose=False`.
2. **Scene**: Added as `jetbot_camera` attribute in `WarehouseSceneCfg`.
3. **Capture**: Use utilities from [camera_utils.py](navisim_lab/utils/camera_utils.py):

```python
from navisim_lab.utils.camera_utils import get_camera_images, save_camera_batch

# Get RGB images as tensor (num_envs, H, W, 3)
rgb = get_camera_images(env.unwrapped, camera_name="jetbot_camera")

# Save all environment camera images to disk
save_camera_batch(env.unwrapped, output_dir="outputs/", camera_name="jetbot_camera", step=0)
```

### RL Training (RSL-RL)

Training uses [RSL-RL](https://github.com/leggedrobotics/rsl_rl) with PPO:

- **Config**: [ppo_warehouse_jetbot.yaml](navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml)
- **Network**: Actor-Critic with 2x256 hidden layers, ELU activation
- **Hyperparameters**: LR=3e-4, GAE lambda=0.95, gamma=0.99, clip=0.2
- **Default**: 10 iterations, 32 steps/env, 4 mini-batches

The environment is wrapped with `RslRlVecEnvWrapper` for RSL-RL compatibility. Note that this wrapper's `step()` returns 4 values `(obs, rewards, dones, info)` instead of Gymnasium's 5-tuple.

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
_setup_scene()          # Create InteractiveScene, clone environments, reset sim
_pre_physics_step()     # Process actions before physics step
_apply_action()         # Write data to simulation (scene.write_data_to_sim())
_get_observations()     # Return {"policy": obs_tensor}
_get_rewards()          # Return reward tensor
_get_dones()            # Return (terminated, truncated) tensors
_reset_idx(env_ids)     # Reset specific environments
```

### Configuration-Driven Design

- `@configclass` decorator for typed configs (`WarehouseEnvCfg`, `WarehouseSceneCfg`)
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

### Camera transform errors ("Prim not xformable with standard transform operations")

This should be handled automatically by `BaseNavigationEnv._standardize_camera_transforms()`. If you still see this error:

1. Clear Python caches: `find . -type d -name __pycache__ -exec rm -rf {} +`
2. Verify your environment inherits from `BaseNavigationEnv`
3. Ensure `_setup_scene()` calls `self._standardize_camera_transforms()` before `self.sim.reset()`

### CUDA architecture errors ("nvrtc: error: invalid value for --gpu-architecture")

On newer GPUs (e.g., Blackwell/GB10, compute capability 10.0) that aren't yet recognized by PyTorch JIT:

```bash
TORCH_CUDA_ARCH_LIST="10.0" PYTORCH_JIT=0 python scripts/rsl_rl/train.py --headless
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

Check USD paths in [paths.py](navisim_lab/utils/paths.py). Verify `ISAAC_NUCLEUS_DIR` points to a valid Isaac Sim nucleus directory.

### Robot falls through ground

Check spawn height in [jetbot_cfg.py](navisim_lab/robots/jetbot_cfg.py). Should be `0.21m` for the 7x-scaled Jetbot (wheel radius 0.03m x 7).

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
