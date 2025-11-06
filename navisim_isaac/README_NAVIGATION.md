# Navigation Agent System for IsaacSim 5.0

Complete system for placing and controlling navigation agents in IsaacSim with NaviSim terrain data.

## Overview

This system provides:
1. **IsaacSimulator**: World management and USD scene loading
2. **NavigationRobot**: Physical robot spawning and control
3. **NavigationController**: Waypoint following and path tracking
4. **TerrainManager**: Heightmap and physics terrain loading
5. **Integration with sector_usd.usd**: Loading NaviSim terrain with Gaussian splatting references

## Quick Start

### Basic Navigation Example

```python
from navisim_isaac.isaac.simulator import IsaacSimulator
from navisim_isaac.isaac.navigation_robot import NavigationRobot
from navisim_isaac.isaac.navigation_controller import NavigationController

# 1. Initialize IsaacSim
simulator = IsaacSimulator()
simulator.initialize()
world = simulator.get_world()

# 2. Spawn robot
robot = NavigationRobot(name="my_robot", world=world)
robot.spawn(position=(0, 0, 0.5))

# 3. Create controller and set waypoints
controller = NavigationController()
controller.set_waypoint_path([
    (2.0, 0.0, 0.5),
    (2.0, 2.0, 0.5),
    (0.0, 0.0, 0.5)
])

# 4. Run simulation
world.reset()
simulator.play()

for _ in range(1000):
    pos, orn = robot.get_pose()
    lin_vel, ang_vel = controller.compute_path_following_velocity(pos, orn)
    robot.set_velocity(lin_vel, ang_vel)
    simulator.step()
```

### Loading Sector USD with Terrain

```python
# Load your sector_usd.usd file
scene_path = simulator.load_sector_usd(
    usd_path="/path/to/sector_usd.usd",
    position=(0.0, 0.0, 0.0)
)

# Set up terrain physics
from navisim_isaac.isaac.terrain import TerrainManager
terrain_manager = TerrainManager(stage=simulator.get_stage())
terrain_manager.load_terrain_from_usd(
    usd_scene_path=scene_path,
    heightmap_prim_name="Heightmap",
    enable_physics=True
)
terrain_manager.set_terrain_properties(friction=0.8, restitution=0.1)
```

## Components

### 1. IsaacSimulator

Main simulation interface:

```python
simulator = IsaacSimulator(config={
    'physics_dt': 1.0/60.0,  # 60 Hz
    'rendering_dt': 1.0/30.0  # 30 Hz
})
simulator.initialize(headless=False)  # Set True for headless mode

# Control simulation
simulator.play()
simulator.pause()
simulator.stop()
simulator.step(num_steps=1)
simulator.reset()
```

### 2. NavigationRobot

Robot spawning and control:

```python
# Create robot
robot = NavigationRobot(
    robot_type="differential_drive",  # or "ackermann", "omnidirectional"
    name="nav_robot",
    world=world
)

# Spawn wheeled robot (Jetbot)
robot.spawn(
    position=(0, 0, 0.5),
    orientation=(0, 0, 0, 1)  # quaternion
)

# Or spawn simple box robot for testing
robot.spawn_simple_robot(
    position=(0, 0, 0.5),
    size=(0.5, 0.5, 0.3)
)

# Control robot
robot.set_velocity(linear_velocity=1.0, angular_velocity=0.5)
robot.set_pose(position=(1, 1, 0.5))  # Teleport
robot.stop()

# Get state
position, orientation = robot.get_pose()
lin_vel, ang_vel = robot.get_velocity()
state = robot.get_state()
```

### 3. NavigationController

Waypoint navigation:

```python
controller = NavigationController(
    max_linear_speed=2.0,      # m/s
    max_angular_speed=2.0,     # rad/s
    position_tolerance=0.1,    # meters
    orientation_tolerance=0.1   # radians
)

# Single waypoint
controller.set_waypoint((5.0, 5.0, 0.5))

# Path of waypoints
controller.set_waypoint_path([
    (2.0, 0.0, 0.5),
    (2.0, 2.0, 0.5),
    (0.0, 2.0, 0.5),
    (0.0, 0.0, 0.5)
])

# Compute velocities for navigation
position, orientation = robot.get_pose()
linear_vel, angular_vel = controller.compute_velocity_to_point(
    current_position=position,
    current_orientation=orientation,
    target_position=(5.0, 5.0, 0.5)
)

# Or follow path automatically
linear_vel, angular_vel = controller.compute_path_following_velocity(
    current_position=position,
    current_orientation=orientation
)
```

### 4. TerrainManager

Terrain and heightmap management:

```python
terrain_manager = TerrainManager(stage=stage)

# Load from USD
terrain_manager.load_terrain_from_usd(
    usd_scene_path="/World/Sector",
    heightmap_prim_name="Heightmap",
    enable_physics=True
)

# Or create from elevation map array
elevation_map = np.random.rand(512, 512) * 2.0  # Random heightmap
terrain_manager.load_height_field(
    elevation_map=elevation_map,
    scale=(1.0, 1.0, 1.0),
    position=(0, 0, 0)
)

# Set physics properties
terrain_manager.set_terrain_properties(
    friction=0.8,              # Surface friction
    restitution=0.1,           # Bounciness
    dynamic_friction=0.7       # Dynamic friction
)

# Query terrain height
height = terrain_manager.get_height_at_position(x=1.0, y=2.0)
```

## Running the Examples

### Full Navigation Example

```bash
cd /Users/jiwon_hae/python_proj/navisim/navisim_isaac
python examples/navigation_example.py
```

This demonstrates:
- Loading sector_usd.usd with heightmap and Gaussian splatting
- Spawning a Jetbot robot
- Navigating a square waypoint path
- Real-time position and velocity feedback

### Simple Test (without USD)

```bash
python examples/navigation_example.py --simple
```

Runs a quick test with simple box robot and single waypoint.

## Sector USD Structure

Your `sector_usd.usd` file should have this structure:

```
World/
├── GaussianScene (references japan.usdz)
│   └── rotation: (-90, 0, 0)
└── Heightmap (Mesh)
    ├── faceVertexCounts: [3, 3, 3, ...]
    ├── faceVertexIndices: [...]
    └── points: [...]
```

The system:
1. Loads the entire USD as a scene reference
2. Applies physics collision to the Heightmap mesh
3. Renders the GaussianScene visually (if IsaacSim supports it)

## Integration with NaviSim SequenceGraph

To integrate with the full NaviSim sequence graph:

```python
from navisim_isaac.navisim.world.sequence_graph import SequenceGraph

# Load sequence graph
seq_graph = SequenceGraph(path="assets/sequence_graph.gpickle")

# Get sector
sequence_id = seq_graph.get_sequence_ids()[0]
sectors = seq_graph.get_sequence(sequence_id)
first_sector = sectors[0]

# Load sector's USD (when available)
sector_usd_path = f"assets/{sequence_id}/{first_sector.sector_id}/sector.usd"
simulator.load_sector_usd(sector_usd_path)

# Access sector data
height_field = first_sector.height_field  # Lazy-loaded
boundary = first_sector.boundary
```

## Coordinate System

- **X-axis**: Forward (robot front direction)
- **Y-axis**: Left (robot left side)
- **Z-axis**: Up (vertical)
- **Orientation**: Quaternion (x, y, z, w)
- **Units**: Meters for distance, radians for angles

## Controller Tuning

Adjust controller gains in `NavigationController`:

```python
controller.k_linear = 1.5   # Higher = faster approach to target
controller.k_angular = 3.0  # Higher = faster rotation to target
```

## Physics Parameters

Adjust simulation fidelity:

```python
config = {
    'physics_dt': 1.0/120.0,  # Higher frequency = more accurate
    'rendering_dt': 1.0/60.0  # Higher frequency = smoother visuals
}
```

## Troubleshooting

### Robot falls through terrain
- Ensure `enable_physics=True` when loading terrain
- Check terrain collision mesh is properly configured
- Increase `physics_dt` frequency

### Robot doesn't move
- Verify robot spawned with `robot.is_spawned`
- Check velocities are being applied: `robot.get_velocity()`
- Ensure simulation is playing: `simulator.is_playing()`

### USD scene not loading
- Verify USD file exists and is valid
- Check file paths are absolute
- Ensure IsaacSim has read permissions

### Navigation overshoots waypoints
- Reduce `max_linear_speed` and `max_angular_speed`
- Increase `position_tolerance` for smoother navigation
- Adjust controller gains (`k_linear`, `k_angular`)

## API Reference

See docstrings in:
- `isaac/simulator.py`
- `isaac/navigation_robot.py`
- `isaac/navigation_controller.py`
- `isaac/terrain.py`

## Next Steps

1. **Add sensors**: Cameras, LiDAR, IMU for perception
2. **Collision detection**: PhysX integration for obstacle detection
3. **Path planning**: A*, RRT for obstacle avoidance
4. **RL training**: Gymnasium environment integration
5. **Multi-robot**: Spawn and control multiple agents

## Requirements

- IsaacSim 5.0
- Python 3.10
- omni.isaac.core
- omni.isaac.wheeled_robots
- numpy

## License

See main repository LICENSE file.
