# Camera Capture in NaviSim IsaacLab

This guide explains how to capture images from the Jetbot's camera during simulation and training.

## Overview

The Jetbot robot has a built-in camera at `chassis/front_cam` in its USD file. You can capture images from this camera in several ways:

1. **Quick snapshots** - Capture periodic images during simulation runs
2. **Camera sensor observations** - Use camera data as RL observations
3. **Video recording** - Record videos during training

---

## Method 1: Quick Snapshots (Simplest)

Use the provided `run_with_snapshots.py` script to capture periodic camera images:

```bash
# Run with default settings (snapshot every 100 steps)
./run_isaac_lab.sh "python scripts/run_with_snapshots.py"

# Custom snapshot interval
./run_isaac_lab.sh "python scripts/run_with_snapshots.py --snapshot_interval 50 --max_steps 500"

# Save to custom directory
./run_isaac_lab.sh "python scripts/run_with_snapshots.py --output_dir outputs/my_snapshots"
```

**Output**: PNG images saved to `outputs/snapshots/` with filenames like `jetbot_env0_step0000.png`

**Use case**: Quick visualization of what the robot sees during navigation

---

## Method 2: Camera Sensor in Scene (For RL Training)

To use camera data as observations in your RL policy, use the camera-enabled scene configuration:

### Step 1: Update Environment Config

```python
# In configs/navigation_env_cfg.py
from scene.warehouse_scene_with_camera_cfg import NavisimWarehouseSceneWithCameraCfg

@configclass
class NavisimNavigationEnvCfg(DirectRLEnvCfg):
    # Replace scene config
    scene: NavisimWarehouseSceneWithCameraCfg = NavisimWarehouseSceneWithCameraCfg(
        num_envs=1,
        env_spacing=4.0,
    )

    # Update observation space to include camera
    # RGB: 480x640x3 = 921,600 dimensions (flatten or use CNN)
    observation_space: int = 13 + 921600  # Root state + RGB pixels
```

### Step 2: Access Camera Data in Environment

```python
# In tasks/navigation_env.py
def _get_observations(self) -> dict:
    """Compute observations including camera data."""
    # Get Jetbot root state
    root_state = self.scene["jetbot"].data.root_state_w.clone()

    # Get camera RGB data
    camera_data = self.scene["front_camera"].data.output["rgb"]  # Shape: (num_envs, H, W, 3)

    # Flatten or process camera data as needed
    camera_flat = camera_data.reshape(self.num_envs, -1)

    # Combine observations
    obs = torch.cat([root_state, camera_flat], dim=-1)

    return {"policy": obs}
```

### Step 3: Save Camera Images During Training

```python
# Add to your training loop
if step % 100 == 0:
    camera_rgb = env.scene["front_camera"].data.output["rgb"][0]  # First env
    # Convert to numpy and save
    from utils.camera_capture import save_camera_image
    save_camera_image(camera_rgb.cpu().numpy(), prefix=f"train_step{step}")
```

**Use case**: Vision-based RL policies (requires CNN or image processing)

---

## Method 3: Video Recording During Training

Your training scripts already support video recording:

```bash
# Train with video recording (uses rgb_array render mode)
./run_isaac_lab.sh "python scripts/train_rsl_rl.py --video"

# Or with Stable Baselines3
./run_isaac_lab.sh "python scripts/train_sb3.py --video"
```

**Output**: Video files saved to the training output directory

**Use case**: Recording agent behavior for analysis and debugging

---

## Camera Configuration Details

### Built-in Jetbot Camera

- **USD Path**: `/World/envs/env_0/Jetbot/chassis/front_cam`
- **Position**: Mounted on chassis front
- **Default Resolution**: 640x480 (configurable)
- **Available Data Types**:
  - `rgb` - RGB color image
  - `distance_to_image_plane` - Depth map
  - `normals` - Surface normals
  - `semantic_segmentation` - Semantic labels (if configured)

### Camera Sensor Configuration

In `warehouse_scene_with_camera_cfg.py`:

```python
front_camera: CameraCfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Jetbot/chassis/front_cam",
    update_period=0.1,  # 10 Hz update rate
    height=480,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=None,  # Don't spawn - use existing USD camera
)
```

**Adjustable Parameters**:
- `update_period`: How often camera captures (seconds)
- `height`, `width`: Image resolution
- `data_types`: What data to capture (RGB, depth, etc.)

---

## Performance Considerations

### Camera Impact on Training Speed

| Configuration | Training Speed | Use Case |
|---------------|----------------|----------|
| No camera | Fastest | State-based policies |
| Camera at 10 Hz | Moderate | Vision policies |
| Camera at 60 Hz | Slower | High-frequency vision |

### Tips for Efficient Camera Use

1. **Lower update rate**: Use `update_period=0.1` (10 Hz) instead of every frame
2. **Reduce resolution**: Use 320x240 instead of 640x480 for faster processing
3. **Headless mode**: Use `--headless` for faster training (no viewer rendering)
4. **GPU tensors**: Keep camera data on GPU to avoid CPU↔GPU transfers

---

## Example: Capture 10 Snapshots

```bash
./run_isaac_lab.sh "python scripts/run_with_snapshots.py \
    --max_steps 1000 \
    --snapshot_interval 100 \
    --output_dir outputs/demo_snapshots"
```

This will create 10 images showing the robot's perspective at different points during navigation.

---

## Troubleshooting

### "Camera not found" error

**Cause**: Jetbot USD file doesn't have the expected camera prim.

**Solution**: Check the actual camera path in the USD file:
```python
# In Isaac Sim, inspect the stage hierarchy
# Or use omni.usd to list prims
```

### Camera images are black/empty

**Cause**: Camera not properly initialized or scene not rendered yet.

**Solution**:
1. Call `camera.initialize()` after environment reset
2. Wait 1-2 simulation steps before capturing first image
3. Ensure `render_mode="human"` or camera update period has elapsed

### Out of memory with camera observations

**Cause**: Camera data (480x640x3 = ~1MB per env) is too large.

**Solution**:
1. Reduce image resolution: `height=240, width=320`
2. Use image encoder/CNN to compress observations
3. Reduce number of parallel environments
4. Process images on GPU to avoid memory copies

---

## Next Steps

- **Vision-based navigation**: Implement CNN-based policy using camera observations
- **Depth sensing**: Use depth data for obstacle avoidance
- **Data collection**: Record camera images for offline learning or dataset creation
- **Multi-camera**: Add additional camera viewpoints for better coverage
