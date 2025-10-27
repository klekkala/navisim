# NaviSim-Isaac Integration

Integration package for NaviSim with Isaac Sim for RL-based autonomous navigation.

## Architecture

This package implements the integration between NaviSim and Isaac Sim following this architecture:

```
NaviSim (SequenceGraph) <-> IsaacSim (Simulation + PhysX) <-> Gymnasium (RL Environment)
```

### Components

#### NaviSim Module (`navisim/`)

**World Components** (`navisim/world/`):
- **SequenceGraph**: NetworkX-based navigation graph loaded from pickle, manages sequences and sectors
- **Sector**: Represents a sector with lazy-loaded height field, USDZ, and boundary polygon data

**Spatial Data Components** (`navisim/spaces/`):
- **HeightField**: Height field data for terrain generation
- **USDZLoader**: USDZ scene file loader for IsaacSim
- **BoundaryPolygon**: Spatial boundary constraints for navigation
- **ElevationMapGenerator**: Generates aligned elevation maps for IsaacSim

**Utilities**:
- **DataConverter**: Converts data between NaviSim and IsaacSim formats

#### IsaacSim Module (`isaac/`)
- **IsaacSimulator**: Main IsaacSim environment interface
- **PhysXCollisionDetector**: Collision detection using PhysX
- **TerrainManager**: Terrain and height field management

#### Gymnasium Module (`gymnasium/`)
- **NaviSimIsaacEnv**: Gymnasium-compatible RL environment
- **NavigationAgent**: Agent interface for navigation
- **RLTrainer**: RL training loop and utilities

## Installation

```bash
# Install in development mode
pip install -e .
```

## Usage

### Basic Example

```python
from navisim_isaac.gymnasium import NaviSimIsaacEnv, NavigationAgent, RLTrainer
from navisim_isaac.config import get_default_config

# Load configuration
config = get_default_config()

# Create environment
env = NaviSimIsaacEnv(config=config["gymnasium"]["env"])

# Create agent
agent = NavigationAgent(config=config["gymnasium"]["agent"])

# Create trainer
trainer = RLTrainer(env=env, agent=agent, config=config["gymnasium"]["training"])

# Train
trainer.train(num_episodes=1000)
```

See `examples/basic_example.py` for a complete example.

## Configuration

Configuration is managed through YAML files. See `config/default_config.yaml` for all available options.

## Project Structure

```
navisim_isaac/
├── __init__.py
├── navisim/              # NaviSim integration
│   ├── __init__.py
│   ├── converters.py          # Data format converters
│   ├── world/                 # Navigation graph structure
│   │   ├── __init__.py
│   │   ├── sequence_graph.py  # NetworkX-based sequence graph
│   │   └── sector.py          # Sector with lazy-loaded data
│   └── spaces/                # Spatial data components
│       ├── __init__.py
│       ├── height_field.py    # Height field data
│       ├── usdz_loader.py     # USDZ scene loader
│       ├── boundary_polygon.py # Boundary polygon data
│       └── elevation_map.py   # Elevation map generation
├── isaac/                # IsaacSim integration
│   ├── simulator.py
│   ├── physx.py
│   └── terrain.py
├── gymnasium/            # RL environment
│   ├── env.py
│   ├── agent.py
│   └── rl.py
├── config/               # Configuration files
│   └── default_config.yaml
├── utils/                # Utilities
│   └── common.py
├── examples/             # Example scripts
│   ├── basic_example.py
│   └── sequence_graph_example.py
└── tests/                # Unit tests
```

## Development Status

This is the initial project structure. Core implementations are in progress.

## License

[Add license information]
