# Navisim

Navisim is a high-fidelity simulation toolkit designed for evaluating autonomous navigation agents in realistic outdoor environments. It leverages real-world data collected via mobile robots and incorporates key features such as elevation map generation, motion data extraction, and environment-aware visual rendering.

> **Note**: The project introduction is subject to change as the final design and scope are still being discussed.

---

## 🚀 Project Introduction

Autonomous agents need to be tested in realistic environments before deployment. **Navisim** addresses this by combining:

- **ROS bag processing** to extract ground-truth control and odometry signals
- **Elevation mapping** from surface point clouds
- **Motion model construction** for trajectory simulation
- Seamless integration with Gymnasium environments to support reinforcement learning workflows

This project enables reproducible experiments and provides a modular backend for developing and benchmarking navigation strategies in complex terrains.
> **Note**: The project introduction is subject to change as the final design and scope are still being discussed.

---

## 🛠 Project Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-org/navisim.git
cd navisim
```

### 2. Create the Conda environment

```bash
conda create -n navisim_env python=3.10
conda activate navisim_env
pip install -r requirements.txt
```

---

## ⚙️ How to Use

### 1. Prepare Required Assets

Download the following files from [this Google Drive link](https://drive.google.com/drive/folders/1mGaTTTblhbVnN_i5AivSQU1ZWyZJIY87?usp=drive_link) and place them inside the `assets/` directory of the project:

- `sequence_graph.gpickle`
- `database.tar`

Then extract the contents of `database.tar`:

```bash
tar -xvf assets/database.tar -C assets/
```

This step sets up the simulation environment with preprocessed point cloud data and graph structures for navigation.

---

### 3. Launch the Navisim Environment

You can now initialize and step through the environment in Python:

```python
from navisim.env import NavisimEnv

env = NavisimEnv(config_path="configs/env_config.yaml")
obs = env.reset()
```

This creates a Gym-compatible environment with access to map data, rendering, and motion simulation.

---

## 📊 Results

Navisim (Beogym) achieves competitive rendering performance when compared to popular simulation platforms. Below is the Frames Per Second (FPS) recorded for each simulator using a single-threaded process at `1280×720` resolution across ~200 timesteps:

| Simulator         | FPS    |
|-------------------|--------|
| Carla             | ~25    |
| Streetlearn       | ~100   |
| Isaac Gym         | ~30    |
| Habitat           | ~370   |
| **Beogym (Ours)** | **~110** |

> **Note**: These results were recorded on a single thread at fixed resolution (1280×720). While Beogym trades off a small amount of performance compared to Habitat, it provides high-quality rendering via Gaussian Splatting and modular real-world integration.

---

## 📁 Folder Structure

```
navisim/
├── gaussian_splatting/     # Integration layer to run Gaussian Splatting rendering components
├── scripts/                        # Standalone scripts for data preprocessing and evaluation
├── src/                              # Core simulation modules
│   ├── config/                 # YAML config files and config parser for environment settings
│   ├── data/                    # Module for loading and querying the simulation database
│   ├── env/                     # Navisim environment setup and Gym-compatible interface
│   ├── motion/               # Motion model and trajectory simulator
│   ├── rendering/           # Camera projection and Gaussian renderer wrapper
│   ├── space/                 # Spatial data structures (poses, transforms, coordinates)
│   ├── utils/                    # Helper functions and utilities used across modules
│   └── enums/                # Enum class definitions for standardizing constant values
├── configs/                       # Experiment and environment configuration files
├── results/                        # Output logs, CSVs, and visual artifacts
└── README.md             # Project documentation
```

---


## ⚡️ Branching Strategy

We follow a structured Git workflow to ensure clarity and maintainability:
- `feat/*` — new feature development
- `fix/*` — bug fixes
- `chore/*` — code cleanup, refactoring, or configuration updates
- `hotfix/*` — urgent patches to production

For more details, refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md).

---

## 📬 Contact

For questions or contributions, please open an issue or contact [email@example.com](mailto\:emaill@example.com)].


