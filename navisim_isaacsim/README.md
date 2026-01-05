# NaviSim Isaac Lab Integration

Isaac Lab integration layer for NaviSim, enabling physics-based reinforcement learning for autonomous navigation with photorealistic rendering.

## Overview

This package (`navisim_isaacsim`) provides a bridge between NaviSim's photorealistic Gaussian Splatting rendering and NVIDIA Isaac Lab's physics simulation and RL workflows. It implements DirectRLEnv-based environments for training navigation agents in warehouse scenarios using the Jetbot robot platform.

## Prerequisites

- **NVIDIA Isaac Sim** (2023.1.0 or later)
- **Isaac Lab** installed and configured
- **NaviSim core package** (parent `navisim/` directory)
- CUDA-capable GPU
- Python 3.10

## Installation

```bash
# From the navisim_isaacsim directory
pip install -e .
```

This installs the `navisim_lab` package which contains all Isaac Lab integration components.

---

## Folder Structure

```
navisim_isaacsim/
├── navisim_lab/                    # Main Python package
│   ├── __init__.py
│   ├── envs/                       # RL Environments
│   │   └── warehouse/              # Warehouse navigation environment
│   │       ├── warehouse_env.py           # DirectRLEnv implementation
│   │       ├── warehouse_env_cfg.py       # Environment configuration (obs/action spaces, rewards)
│   │       └── warehouse_scene_cfg.py     # Scene layout (assets, lighting, robot spawn)
│   ├── robots/                     # Robot configurations
│   │   └── jetbot_cfg.py          # Jetbot ArticulationCfg with wheel actuators
│   ├── camera/                     # Camera system
│   │   └── jetbot_camera.py       # POV camera configuration for Jetbot
│   ├── tasks/                      # Task registry
│   │   ├── __init__.py            # Imports warehouse tasks for registration
│   │   └── warehouse/             # Warehouse task definitions
│   │       ├── __init__.py        # Registers "Navisim-Warehouse-Jetbot-v0"
│   │       └── agents/            # Agent-specific configurations
│   ├── configs/                    # Training configurations
│   │   └── rsl_rl/                # RSL-RL (PPO) configs
│   │       ├── ppo_cfg.py                # Config loader for RSL-RL
│   │       └── ppo_warehouse_jetbot.yaml # PPO hyperparameters
│   └── utils/                      # Utility modules
│       ├── paths.py               # Isaac Nucleus asset paths (warehouse/jetbot USD)
│       └── camera_capture.py      # Camera frame extraction utilities
├── scripts/                        # Execution scripts
│   ├── smoke_test.py              # Quick integration test
│   ├── run_with_jetbot_camera.py  # Run environment with camera capture
│   └── rsl_rl/                    # RL training scripts
│       ├── train.py               # Train PPO agent
│       └── play.py                # Evaluate trained agent
├── outputs/                        # Generated outputs
│   └── jetbot_pov/                # Captured camera frames
├── pyproject.toml                  # Package metadata
└── README.md                       # This file
```

---

## Component Roles

### 1. Environments (`navisim_lab/envs/`)

The core RL environment implementations using Isaac Lab's DirectRLEnv pattern.

#### **WarehouseEnv** ([warehouse_env.py](navisim_lab/envs/warehouse/warehouse_env.py))
- **Purpose**: Main RL environment for Jetbot navigation in warehouse
- **Base Class**: `DirectRLEnv` (Isaac Lab's optimized RL interface)
- **Key Methods**:
  - `_setup_scene()`: Creates InteractiveScene, clones environments
  - `_pre_physics_step(actions)`: Applies wheel velocity commands
  - `_get_observations()`: Returns 13D state (position, quaternion, velocities)
  - `_get_rewards()`: Computes forward progress reward
- **Observations**: `[pos(3), quat(4), lin_vel(3), ang_vel(3)]` = 13D
- **Actions**: `[left_wheel_vel, right_wheel_vel]` = 2D (normalized [-1, 1])

#### **WarehouseEnvCfg** ([warehouse_env_cfg.py](navisim_lab/envs/warehouse/warehouse_env_cfg.py))
- **Purpose**: Configuration for environment parameters
- **Base Class**: `DirectRLEnvCfg`
- **Key Settings**:
  - Simulation: 60 Hz physics, CUDA device
  - Episode length: 10 seconds
  - Action scaling: 5.0 rad/s for wheel velocities
  - Reward weights: Forward progress weight = 1.0
  - Decimation: 2 (sim steps per env step)

#### **WarehouseSceneCfg** ([warehouse_scene_cfg.py](navisim_lab/envs/warehouse/warehouse_scene_cfg.py))
- **Purpose**: Defines scene layout and asset spawning
- **Base Class**: `InteractiveSceneCfg`
- **Assets**:
  - `warehouse`: Static warehouse USD asset
  - `jetbot`: Per-environment cloned robot with actuators
  - `jetbot_camera`: POV camera reference (existing in Jetbot USD)
- **Pattern**: Declarative asset definitions using class attributes

---

### 2. Robots (`navisim_lab/robots/`)

Robot configuration definitions using Isaac Lab's Articulation system.

#### **JETBOT_CONFIG** ([jetbot_cfg.py](navisim_lab/robots/jetbot_cfg.py))
- **Purpose**: Jetbot robot articulation configuration
- **Key Features**:
  - USD file spawning with 7x scale (human-sized for better physics)
  - Initial spawn height: 0.21m (wheels on ground)
  - Actuators: `ImplicitActuatorCfg` for wheel velocity control
    - Joint names: `left_wheel_joint`, `right_wheel_joint`
    - High damping (1000.0) for scaled robot mass
    - Velocity limit: 200 rad/s
    - Effort limit: 5000 N·m (scaled for 7³ mass increase)

---

### 3. Camera (`navisim_lab/camera/`)

Camera system for POV rendering and observation capture.

#### **jetbot_pov_camera** ([jetbot_camera.py](navisim_lab/camera/jetbot_camera.py))
- **Purpose**: First-person camera attached to Jetbot
- **Type**: `CameraCfg` (Isaac Lab sensor)
- **Key Features**:
  - References existing camera in Jetbot USD (spawn=None)
  - Path: `{ENV_REGEX_NS}/Jetbot/chassis/rgb_camera/jetbot_camera`
  - Resolution: 640x480 RGB
  - `update_latest_camera_pose=True`: Tracks robot movement via XFormPrimView
- **Use Case**: Visual observations, debugging, camera-based RL policies

---

### 4. Tasks (`navisim_lab/tasks/`)

Task registry system for Gymnasium environment registration.

#### **Purpose**
Registers Isaac Lab environments with Gymnasium using the standard `gym.register()` pattern.

#### **Structure**
```python
# tasks/__init__.py imports all task subpackages
from .warehouse import *

# tasks/warehouse/__init__.py registers the environment
gymnasium.register(
    id="Navisim-Warehouse-Jetbot-v0",
    entry_point="navisim_lab.envs.warehouse:WarehouseEnv",
    env_cfg_entry_point="navisim_lab.envs.warehouse:WarehouseEnvCfg",
)
```

#### **Usage**
```python
import gymnasium as gym
import navisim_lab.tasks  # Trigger registration

env = gym.make("Navisim-Warehouse-Jetbot-v0", cfg=env_cfg)
```

---

### 5. Configs (`navisim_lab/configs/`)

Training algorithm configurations for RL workflows.

#### **RSL-RL PPO Configuration** ([rsl_rl/](navisim_lab/configs/rsl_rl/))

**ppo_cfg.py**:
- Loads YAML configuration into RSL-RL config objects
- Creates nested structure: `RslRlOnPolicyRunnerCfg` → `RslRlPpoActorCriticCfg` + `RslRlPpoAlgorithmCfg`
- Exposes `PPO_WAREHOUSE_JETBOT_CFG` instance

**ppo_warehouse_jetbot.yaml**:
- PPO hyperparameters (learning rate, batch size, GAE lambda, etc.)
- Actor-critic network architecture (hidden layers, activation functions)
- Training schedule (max iterations, save interval)

---

### 6. Utils (`navisim_lab/utils/`)

Utility modules for paths and camera operations.

#### **paths.py**
- **Purpose**: Centralized asset path management
- **Exports**:
  - `WAREHOUSE_USD`: Path to warehouse environment USD file
  - `JETBOT_USD`: Path to Jetbot robot USD file
  - Uses `ISAAC_NUCLEUS_DIR` for Isaac Sim built-in assets

#### **camera_capture.py**
- **Purpose**: Utilities for extracting and saving camera frames
- **Use Case**: Recording POV videos, generating datasets from simulation

---

### 7. Scripts (`scripts/`)

Executable scripts for running environments and training agents.

#### **smoke_test.py**
- **Purpose**: Quick integration test for environment setup
- **What it does**:
  - Launches Isaac Sim with AppLauncher
  - Creates `Navisim-Warehouse-Jetbot-v0` environment
  - Runs 10 random action steps
  - Validates environment lifecycle (reset, step, close)

#### **run_with_jetbot_camera.py**
- **Purpose**: Run environment with camera frame capture
- **Features**:
  - Records POV images from Jetbot camera
  - Saves frames to `outputs/jetbot_pov/`
  - Supports headless mode for server deployments
  - Configurable number of environments and episode length

#### **rsl_rl/train.py** (if exists)
- **Purpose**: Train PPO agent using RSL-RL
- **Features**:
  - Loads PPO config from YAML
  - Parallelized training across multiple environments
  - Saves checkpoints to logs directory

#### **rsl_rl/play.py** (if exists)
- **Purpose**: Evaluate trained agent
- **Features**:
  - Loads trained policy checkpoint
  - Runs deterministic policy for evaluation
  - Visualizes agent behavior in Isaac Sim viewer

---

## Usage

### Docker Setup (Recommended)

If running Isaac Lab in a Docker container, ensure GPU access is properly configured:

```bash
# Enable X11 forwarding (for GUI mode)
xhost +local:docker

# Run Isaac Lab container with GPU support
docker run -it --rm \
    --gpus all \
    --network host \
    --ipc host \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=$XAUTHORITY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $XAUTHORITY:$XAUTHORITY \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data \
    -v ~/docker/isaac-sim/documents:/root/Documents \
    -v "$(pwd)":/workspace/user \
    --workdir /workspace/user \
    nvcr.io/nvidia/isaac-lab:2.3.0

# Inside container, verify GPU access
nvidia-smi  # Should show GPU info
```

**Important Docker Notes**:
- `--gpus all` is **required** for CUDA access
- Remove `-e EGL_PLATFORM=surfaceless` if it causes CUDA initialization issues
- For headless mode, use `--headless` flag in Python scripts instead of container env vars
- X11 authorization (`XAUTHORITY`) may be needed even for headless mode to initialize GPU properly

### Quick Start

```bash
# 1. Run integration test
cd navisim_isaacsim
python scripts/smoke_test.py

# 2. Run smoke test in headless mode (recommended for containers)
python scripts/smoke_test.py --headless

# 3. Run with camera capture
python scripts/run_with_jetbot_camera.py --num_envs 1

# 4. Train RL agent
python scripts/rsl_rl/train.py --num_envs 64 --headless

# 5. Evaluate trained agent
python scripts/rsl_rl/play.py --checkpoint logs/rsl_rl/warehouse_jetbot/ppo/model_10.pt --headless
```

### Creating Custom Environments

1. **Define scene config** in `navisim_lab/envs/<your_env>/`:
   ```python
   @configclass
   class MySceneCfg(InteractiveSceneCfg):
       robot = MY_ROBOT_CONFIG
       # Add your assets...
   ```

2. **Implement environment** inheriting from `DirectRLEnv`:
   ```python
   class MyEnv(DirectRLEnv):
       def _setup_scene(self): ...
       def _get_observations(self): ...
       def _get_rewards(self): ...
       def _pre_physics_step(self, actions): ...
   ```

3. **Register with Gymnasium** in `navisim_lab/tasks/<your_env>/`:
   ```python
   gymnasium.register(
       id="MyTask-v0",
       entry_point="navisim_lab.envs.<your_env>:MyEnv",
       env_cfg_entry_point="navisim_lab.envs.<your_env>:MyEnvCfg",
   )
   ```

### Adding New Robots

1. Create `navisim_lab/robots/<robot_name>_cfg.py`:
   ```python
   ROBOT_CONFIG = ArticulationCfg(
       prim_path="{ENV_REGEX_NS}/Robot",
       spawn=sim_utils.UsdFileCfg(usd_path=ROBOT_USD),
       actuators={...},
   )
   ```

2. Reference in scene config:
   ```python
   from navisim_lab.robots.<robot_name>_cfg import ROBOT_CONFIG

   class MySceneCfg(InteractiveSceneCfg):
       robot = ROBOT_CONFIG
   ```

---

## Key Design Patterns

### 1. **DirectRLEnv Pattern**
- Single-class implementation for transparent, optimized RL
- All logic in environment class (vs. Manager-Based approach)
- Methods: `_setup_scene()`, `_pre_physics_step()`, `_get_observations()`, `_get_rewards()`
- GPU tensors for all state/action/reward data

### 2. **InteractiveScene Management**
- Handles multi-environment cloning automatically
- Dictionary-style asset access: `self.scene["jetbot"]`
- `env_origins` for spatial offsetting of parallel environments
- Must call `scene.write_data_to_sim()` to apply actions

### 3. **Configuration-Driven Design**
- `@configclass` decorator for typed configurations
- Separate config classes for scene, environment, and training
- YAML files for hyperparameters (easy tuning without code changes)

### 4. **AppLauncher Pattern**
- **CRITICAL**: Must launch Isaac Sim before importing Omniverse modules
- Always first two lines of scripts:
  ```python
  from isaaclab.app import AppLauncher
  app_launcher = AppLauncher(headless=False)
  simulation_app = app_launcher.app
  ```

---

## Integration with NaviSim Core

This package is designed to work alongside the core `navisim` package:

- **NaviSim Core** ([../navisim/](../navisim/)): Gaussian Splatting rendering, spatial data structures, ROS bag processing
- **NaviSim Isaac Lab** (this package): Physics simulation, RL environments, robot control

### Workflow
1. Use NaviSim core to process real-world data (elevation maps, Gaussian models)
2. Use Isaac Lab integration for training navigation policies
3. Policies can be tested with NaviSim's photorealistic rendering
4. Deploy trained policies on physical robots

---

## Troubleshooting

### **CUDA Not Available / "No CUDA GPUs are available"**

This is the most common issue when running Isaac Sim in Docker containers.

**Symptoms**:
```
[Error] [omni.physx.plugin] CUDA libs are present, but no suitable CUDA GPU was found!
[Error] [carb.cudainterop.plugin] CUDA error 100: cudaErrorNoDevice
RuntimeError: No CUDA GPUs are available
```

**Root Causes**:
1. Container launched without `--gpus all` flag
2. X11 authorization issues preventing GPU initialization
3. `EGL_PLATFORM=surfaceless` interfering with CUDA context

**Solutions** (try in order):

1. **Verify GPU access in container**:
   ```bash
   # Inside container, run:
   nvidia-smi  # Should display GPU info
   ls -la /dev/nvidia*  # Should show device nodes
   ```

2. **Fix Docker launch command**:
   ```bash
   # Remove -e EGL_PLATFORM=surfaceless
   # Add XAUTHORITY mount
   docker run -it --rm \
       --gpus all \
       -e DISPLAY=$DISPLAY \
       -e XAUTHORITY=$XAUTHORITY \
       -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
       -v $XAUTHORITY:$XAUTHORITY \
       ...
   ```

3. **Use headless mode**:
   ```bash
   # Always add --headless flag to scripts when running in container
   python scripts/rsl_rl/train.py --headless
   python scripts/rsl_rl/play.py --headless --checkpoint model.pt
   ```

4. **Check if Isaac Sim's PyTorch sees CUDA**:
   ```bash
   # Inside container, after AppLauncher:
   python -c "from isaaclab.app import AppLauncher; app = AppLauncher(); import torch; print(torch.cuda.is_available())"
   ```

5. **Fallback to CPU mode** (performance degradation expected):
   - The scripts automatically detect CUDA unavailability
   - Environment will use `device="cpu"` and policy will load on CPU
   - Physics simulation will be slower but functional

**Known Issue**: Isaac Sim 2024.x on some systems requires X11 authentication even for headless GPU mode. The `XAUTHORITY` mount is essential.

### **"Module 'isaaclab' not found"**
Ensure Isaac Lab is installed and `PYTHONPATH` includes Isaac Lab directory.

### **AppLauncher import errors**
Always import and launch AppLauncher before any Isaac/Omniverse modules.

### **"Authorization required, but no authorization protocol specified"**
X11 authentication issue. Run `xhost +local:docker` before launching container, or mount `$XAUTHORITY` file.

### **Scene assets not appearing**
Check USD paths in `navisim_lab/utils/paths.py`. Verify `ISAAC_NUCLEUS_DIR` points to valid Isaac Sim installation.

### **Camera not tracking robot movement**
Ensure camera config has `update_latest_camera_pose=True` and `spawn=None` (to reference existing camera in USD).

### **PhysX warnings on startup**
Normal. Ensure `sim.reset()` is called after creating InteractiveScene.

### **Robot falls through ground**
Check robot spawn height in ArticulationCfg InitialStateCfg. Should account for wheel/base height.

### **"ValueError: not enough values to unpack (expected 5, got 4)"**
`RslRlVecEnvWrapper.step()` returns 4 values (obs, rewards, dones, info), not standard Gymnasium 5-tuple. Use:
```python
obs, rewards, dones, info = env.step(actions)  # Correct for wrapped env
```

---

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab)
- [Isaac Lab DirectRLEnv Guide](https://isaac-sim.github.io/IsaacLab/source/tutorials/03_envs/create_direct_rl_env.html)
- [NaviSim Core Package](../navisim/)
- [RSL-RL Repository](https://github.com/leggedrobotics/rsl_rl)

---

## Contributing

Follow the branch naming convention:
- `feat/*` — new features
- `fix/*` — bug fixes
- `chore/*` — refactoring, configuration, cleanup

Main branch: `main`

See [CONTRIBUTING.md](../.github/CONTRIBUTING.md) for full guidelines.
