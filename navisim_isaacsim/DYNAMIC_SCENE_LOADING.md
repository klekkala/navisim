## Dynamic Scene Loading for NaviSim

This document describes the dynamic USD section loading system that enables efficient navigation through large-scale environments in Isaac Sim.

## Overview

The dynamic scene loading system allows you to:
- **Stream USD sections** as robots navigate through large maps
- **Use NetworkX graphs** to represent spatial connectivity
- **Automatically load/unload** sections based on robot position
- **Manage memory efficiently** by limiting loaded sections
- **Handle section transitions** smoothly during navigation

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Your Environment                        │
│              (WarehouseEnv, etc.)                       │
└───────────────────┬─────────────────────────────────────┘
                    │ Wrapped by
┌───────────────────▼─────────────────────────────────────┐
│            DynamicSceneEnvWrapper                        │
│  - Tracks robot position                                │
│  - Triggers scene updates                               │
│  - Adds scene info to env state                         │
└───────────────────┬─────────────────────────────────────┘
                    │ Uses
┌───────────────────▼─────────────────────────────────────┐
│          DynamicSceneManager                             │
│  - Loads/unloads USD sections                           │
│  - Manages Isaac Sim stage                              │
│  - Tracks loading state                                 │
└───────────────────┬─────────────────────────────────────┘
                    │ Uses
┌───────────────────▼─────────────────────────────────────┐
│              SceneGraph                                  │
│  - NetworkX graph of sections                           │
│  - Spatial queries (nearest, radius, path)              │
│  - Section metadata                                     │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Prepare Your Data

Convert your map sections to USD files and organize them:

```
assets/
├── sections/
│   ├── section_001.usd
│   ├── section_002.usd
│   ├── section_003.usd
│   └── ...
└── scene_graph.pkl  # NetworkX graph with section metadata
```

### 2. Build Scene Graph

Create a scene graph from your section data:

```python
from navisim_lab.scene import SceneGraph, build_scene_graph_from_data
import numpy as np

# Define your sections
sections_data = [
    {
        'id': 'section_001',
        'usd_path': 'assets/sections/section_001.usd',
        'bounds': [[0, 0, 0], [50, 50, 10]],  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        'center': [25, 25, 5],  # [x, y, z]
        'metadata': {'type': 'warehouse', 'difficulty': 'easy'},
        'neighbors': ['section_002', 'section_003']  # Connected sections
    },
    {
        'id': 'section_002',
        'usd_path': 'assets/sections/section_002.usd',
        'bounds': [[50, 0, 0], [100, 50, 10]],
        'center': [75, 25, 5],
        'neighbors': ['section_001', 'section_004']
    },
    # ... more sections
]

# Build scene graph
from navisim_lab.scene.build_scene_graph import build_scene_graph_from_data
scene_graph = build_scene_graph_from_data(sections_data)

# Save for reuse
scene_graph.to_pickle('assets/scene_graph.pkl')

# Visualize
scene_graph.visualize('scene_graph.png')
```

### 3. Use with Environment

Wrap your environment to enable dynamic loading:

```python
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from navisim_lab.scene import SceneGraph, DynamicSceneEnvWrapper

# Create base environment
env_cfg = parse_env_cfg("Navisim-Warehouse-Jetbot-v0", num_envs=1)
env = gym.make("Navisim-Warehouse-Jetbot-v0", cfg=env_cfg)

# Load scene graph
scene_graph = SceneGraph.from_pickle("assets/scene_graph.pkl")

# Wrap environment
env = DynamicSceneEnvWrapper(
    env,
    scene_graph=scene_graph,
    max_loaded_sections=9,        # Keep at most 9 sections loaded
    load_radius=50.0,             # Load sections within 50m
    neighbor_depth=1,             # Load immediate neighbors
    update_frequency=10,          # Update every 10 steps
    preload_sections=['section_001']  # Initial sections to load
)

# Use normally
obs, info = env.reset()

for step in range(1000):
    action = policy(obs)
    obs, reward, done, info = env.step(action)

    # Check scene info
    if 'scene_update' in info:
        print(f"Loaded: {info['scene_update']['loaded']}")
        print(f"Unloaded: {info['scene_update']['unloaded']}")

    if 'section_transition' in info:
        print(f"Robot moved to section: {info['current_section']}")
```

## Core Components

### SceneSection

Represents a single USD section:

```python
from navisim_lab.scene import SceneSection
import numpy as np

section = SceneSection(
    section_id="warehouse_01",
    usd_path="assets/sections/warehouse_01.usd",
    bounds=np.array([[0, 0, 0], [50, 50, 10]]),
    center=np.array([25, 25, 5]),
    metadata={'type': 'warehouse', 'objects': 150}
)

# Check if point is in section
point = np.array([30, 20, 5])
is_inside = section.contains_point(point)  # True

# Distance to section
distance = section.distance_to_point(np.array([100, 100, 5]))
```

### SceneGraph

Manages the spatial graph of sections:

```python
from navisim_lab.scene import SceneGraph

# Load from pickle
scene_graph = SceneGraph.from_pickle("scene_graph.pkl")

# Query operations
section_id = scene_graph.find_section_containing_point([25, 25, 5])

# Get neighbors
neighbors = scene_graph.get_neighbors("section_001", depth=2)

# Find K nearest sections
nearest = scene_graph.get_k_nearest_sections([100, 100, 5], k=5)
# Returns: [('section_X', distance), ...]

# Get sections within radius
nearby = scene_graph.get_sections_within_radius([50, 50, 5], radius=30.0)

# Compute loading strategy (what to load/unload)
to_load, to_unload = scene_graph.compute_loading_strategy(
    current_position=[25, 25, 5],
    max_loaded_sections=9,
    load_radius=50.0,
    neighbor_depth=1
)

# Path planning through sections
path = scene_graph.get_path_sections("section_001", "section_050")
# Returns: ['section_001', 'section_002', ..., 'section_050']

# Visualize
scene_graph.visualize('graph.png')
```

### DynamicSceneManager

Handles USD loading/unloading:

```python
from navisim_lab.scene import DynamicSceneManager, SceneGraph

scene_graph = SceneGraph.from_pickle("scene_graph.pkl")
stage = simulation_app.context.get_stage()

manager = DynamicSceneManager(
    scene_graph=scene_graph,
    stage=stage,
    max_loaded_sections=9,
    load_radius=50.0,
    neighbor_depth=1
)

# Manual loading
manager.load_section("section_001")
manager.unload_section("section_002")

# Automatic update based on robot position
import numpy as np
robot_pos = np.array([25, 25, 5])
result = manager.update(robot_pos)
# Returns: {'loaded': ['section_X', ...], 'unloaded': ['section_Y', ...]}

# Preload a path
path = scene_graph.get_path_sections("start", "goal")
manager.preload_path(path)

# Get statistics
stats = manager.get_stats()
# Returns: {
#     'total_sections': 100,
#     'loaded_sections': 9,
#     'total_loads': 45,
#     'total_unloads': 36,
#     ...
# }
```

### DynamicSceneEnvWrapper

Integrates with IsaacLab environments:

```python
from navisim_lab.scene import DynamicSceneEnvWrapper

# Wrap environment
env = DynamicSceneEnvWrapper(
    env,
    scene_graph=scene_graph,
    max_loaded_sections=9,
    load_radius=50.0,
    neighbor_depth=1,
    update_frequency=10,  # Update every 10 steps
    robot_index=0,        # Track first robot in multi-robot envs
    preload_sections=['section_001', 'section_002']
)

# Environment step automatically handles scene updates
obs, reward, done, info = env.step(action)

# Access scene information
current_section = info.get('current_section')
scene_stats = info.get('scene_stats')

if 'scene_update' in info:
    print(f"Loaded: {info['scene_update']['loaded']}")
    print(f"Unloaded: {info['scene_update']['unloaded']}")

if 'section_transition' in info:
    print(f"Transitioned from {info['section_transition']['from']} "
          f"to {info['section_transition']['to']}")

# Preload a specific path
manager.preload_path(['section_001', 'section_002', 'section_003'])

# Get statistics
stats = env.get_scene_stats()

# Visualize graph
env.visualize_scene_graph('graph_visualization.png')
```

## Configuration Guidelines

### Section Size

Choose section sizes based on:
- **Too small**: Frequent loading/unloading overhead
- **Too large**: Memory issues, long load times
- **Recommended**: 50m x 50m to 100m x 100m per section

### Loading Parameters

**max_loaded_sections**:
- Depends on available VRAM
- Typical: 9-16 sections for 24GB VRAM
- Monitor: Use `get_stats()` to track memory usage

**load_radius**:
- Should cover robot's sensor range + safety margin
- Typical: 50-100m for warehouse environments
- Formula: `load_radius = sensor_range + robot_speed * update_interval + margin`

**neighbor_depth**:
- 1: Load only adjacent sections (faster, less memory)
- 2: Load neighbors of neighbors (smoother transitions)
- 3+: Rarely needed unless very fast robots

**update_frequency**:
- Lower = more responsive, higher overhead
- Higher = less overhead, potential gaps
- Typical: 10-30 steps depending on robot speed
- Formula: `frequency = (section_size / robot_speed) / sim_dt`

## Advanced Usage

### Custom Loading Strategy

Override the loading strategy for custom behavior:

```python
class CustomSceneManager(DynamicSceneManager):
    def compute_loading_strategy(self, robot_position):
        # Custom logic: prioritize sections in robot's heading direction

        # Get robot heading from env
        heading = self.get_robot_heading()

        # Score sections by alignment with heading
        sections_scored = []
        for section_id, section in self.scene_graph.sections.items():
            to_section = section.center - robot_position
            alignment = np.dot(to_section / np.linalg.norm(to_section), heading)
            sections_scored.append((section_id, alignment))

        # Load top-scored sections
        sections_scored.sort(key=lambda x: x[1], reverse=True)
        to_load = {sid for sid, _ in sections_scored[:self.max_loaded_sections]}
        to_unload = self.loaded_sections - to_load

        return to_load, to_unload
```

### Multi-Robot Support

Handle multiple robots tracking different sections:

```python
# Create separate wrappers for each robot
wrappers = []
for robot_idx in range(num_robots):
    wrapper = DynamicSceneEnvWrapper(
        env,
        scene_graph=scene_graph,
        robot_index=robot_idx,
        max_loaded_sections=9,  # Shared budget
        update_frequency=10
    )
    wrappers.append(wrapper)

# Coordinate loading across robots
loaded_union = set()
for wrapper in wrappers:
    loaded_union.update(wrapper.scene_manager.loaded_sections)

# Ensure total doesn't exceed limit
if len(loaded_union) > total_section_budget:
    # Unload least important sections
    ...
```

### Section Metadata

Use metadata for intelligent loading:

```python
# Add metadata when building graph
sections_data = [
    {
        'id': 'warehouse_01',
        'usd_path': 'assets/sections/warehouse_01.usd',
        'bounds': [[0, 0, 0], [50, 50, 10]],
        'center': [25, 25, 5],
        'metadata': {
            'type': 'warehouse',
            'difficulty': 'easy',
            'object_density': 'high',
            'priority': 1,
            'lighting': 'bright'
        }
    }
]

# Query based on metadata
high_priority = [
    sid for sid, section in scene_graph.sections.items()
    if section.metadata.get('priority', 0) > 5
]

# Preload high-priority sections
manager.preload_path(high_priority)
```

### Performance Monitoring

Track and optimize loading performance:

```python
import time

class MonitoredSceneManager(DynamicSceneManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_times = []
        self.unload_times = []

    def load_section(self, section_id):
        start = time.time()
        result = super().load_section(section_id)
        elapsed = time.time() - start
        self.load_times.append(elapsed)

        if elapsed > 0.1:  # Warn if slow
            logger.warning(f"Slow load: {section_id} took {elapsed:.3f}s")

        return result

    def get_performance_stats(self):
        return {
            'avg_load_time': np.mean(self.load_times),
            'max_load_time': np.max(self.load_times),
            'avg_unload_time': np.mean(self.unload_times),
        }
```

## Troubleshooting

### Sections Not Loading

**Symptoms**: Sections don't appear in simulation

**Solutions**:
- Check USD file paths are correct
- Verify bounds/center coordinates are accurate
- Ensure section IDs match between graph and USD files
- Check stage reference in DynamicSceneManager

```python
# Debug loading
manager = DynamicSceneManager(scene_graph, stage, ...)
result = manager.load_section("section_001")
if not result:
    logger.error("Failed to load section_001")
    section = scene_graph.get_section("section_001")
    print(f"USD path: {section.usd_path}")
    print(f"Exists: {section.usd_path.exists()}")
```

### Robot Falls Through Geometry

**Symptoms**: Robot falls when transitioning sections

**Solutions**:
- Ensure section bounds overlap slightly (1-2m)
- Increase `load_radius` to preload earlier
- Reduce `update_frequency` for faster updates
- Add buffer zones between sections

### High Memory Usage

**Symptoms**: VRAM exhausted, Isaac Sim crashes

**Solutions**:
- Reduce `max_loaded_sections`
- Optimize USD files (reduce polygon count)
- Decrease `neighbor_depth`
- Use Level of Detail (LOD) in USD files

```python
# Monitor memory
stats = manager.get_stats()
if stats['loaded_sections'] > 10:
    logger.warning(f"High section count: {stats['loaded_sections']}")
```

### Frequent Loading/Unloading

**Symptoms**: Stuttering, performance drops

**Solutions**:
- Increase `load_radius` for more buffer
- Increase `update_frequency` (update less often)
- Add hysteresis to loading strategy
- Increase section sizes

## Integration with Training

### Example: PPO Training with Dynamic Scenes

```python
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from navisim_lab.scene import SceneGraph, DynamicSceneEnvWrapper
from navisim_lab.agents import create_agent

# Create environment
env = gym.make("Navisim-Warehouse-Jetbot-v0", cfg=env_cfg)

# Add dynamic scene loading
scene_graph = SceneGraph.from_pickle("assets/scene_graph.pkl")
env = DynamicSceneEnvWrapper(
    env,
    scene_graph=scene_graph,
    max_loaded_sections=9,
    update_frequency=30  # Less frequent for training speed
)

# Wrap for RL
agent = create_agent("rsl_rl", algorithm="ppo")
env = agent.wrap_env(env)

# Train
agent.setup(env, config)
agent.train(num_iterations=1000, log_dir="logs/")
```

## Best Practices

1. **Test with small graphs first**: Debug with 5-10 sections before scaling
2. **Profile section sizes**: Measure load times, optimize USD files
3. **Use metadata**: Encode important info in section metadata
4. **Monitor statistics**: Track loaded sections, transitions
5. **Visualize graphs**: Use `visualize()` to debug connectivity
6. **Overlap sections**: 1-2m overlap prevents gaps
7. **Preload paths**: For known routes, preload ahead of time
8. **Cache graphs**: Save built graphs with `to_pickle()`

## Future Enhancements

Planned features:
- [ ] Asynchronous loading (load in background thread)
- [ ] Predictive loading (based on robot trajectory)
- [ ] Level of Detail (LOD) support
- [ ] Dynamic object streaming within sections
- [ ] Multi-GPU section distribution
- [ ] Compression for faster loading

## Examples

See `examples/dynamic_scene_loading/` for complete examples:
- `basic_usage.py`: Simple dynamic loading setup
- `path_preloading.py`: Preload sections along a route
- `multi_robot.py`: Coordinate loading across multiple robots
- `custom_strategy.py`: Implement custom loading logic

## Support

For questions or issues:
- Check this documentation
- Review examples in `examples/dynamic_scene_loading/`
- File an issue with "dynamic-scene" label

---

**Ready to use dynamic scene loading!**

```bash
python scripts/train.py \
    --task Navisim-Warehouse-Jetbot-v0 \
    --agent rsl_rl \
    --config config.yaml \
    # Environment will automatically stream USD sections as robot navigates
```
