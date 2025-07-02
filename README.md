# Navisim

Navisim is a high-fidelity simulation toolkit designed for evaluating autonomous navigation agents in realistic outdoor environments. It leverages real-world data collected via mobile robots and incorporates key features such as elevation map generation, motion data extraction, and environment-aware visual rendering.

> **Note**: The project introduction is subject to change as the final design and scope are still being discussed.

---

## 🚀 Project Introduction

Autonomous agents need to be tested in realistic environments before deployment. **Navisim** addresses this by combining:

- **ROS bag processing** to extract ground-truth control and odometry signals
- **Elevation mapping** from surface point clouds
- **Motion model construction** for trajectory simulation
- **Optimized C++ backend** for high-performance Gaussian Splatting rendering
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
```

### 3. Install dependencies and build C++ extensions

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install build dependencies for C++ extensions
pip install pybind11

# Build and install the package with C++ optimizations
pip install -e .
```

**Alternative development build** (for active C++ development):
```bash
python setup.py build_ext --inplace
```

### 4. Verify installation

Test that the C++ extensions are working:

```bash
python -c "import navisim.gaussian_ops; print('C++ extensions loaded successfully!')"
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

## 🔧 Development

### Working with C++ Extensions

Navisim includes optimized C++ components for performance-critical operations like Gaussian Splatting rendering. These are located in:

```
navisim/
├── _native/                    # C++ source files
│   ├── gaussian_ops.cpp       # Main Gaussian Splatting operations
│   ├── gaussian_rasterization.cpp
│   └── gaussian_rasterization_cuda.cu  # CUDA kernels (if available)
```

**After modifying C++ code:**

1. Rebuild the extensions:
   ```bash
   python setup.py build_ext --inplace
   ```

2. Test your changes:
   ```bash
   python test_module.py
   ```

**Requirements:**
- C++14 or higher compatible compiler
- CUDA toolkit (optional, for GPU acceleration)
- PyTorch (for tensor operations)

### Building for Distribution

To create wheels for distribution:

```bash
# Build for current platform
python setup.py bdist_wheel

# Build source distribution
python setup.py sdist
```

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

> **Note**: These results were recorded on a single thread at fixed resolution (1280×720). While Beogym trades off a small amount of performance compared to Habitat, it provides high-quality rendering via Gaussian Splatting and modular real-world integration. Performance may improve significantly with CUDA-enabled builds.

---

## 📁 Folder Structure

```
navisim/
├── _native/                         # C++ extensions for performance optimization
│   ├── gaussian_ops.cpp            # Core Gaussian Splatting operations
│   └── gaussian_rasterization_cuda.cu  # CUDA kernels for GPU acceleration
├── gaussian_splatting/              # Integration layer to run Gaussian Splatting rendering components
├── scripts/                         # Standalone scripts for data preprocessing and evaluation
├── src/                             # Core simulation modules
│   ├── config/                      # YAML config files and config parser for environment settings
│   ├── data/                        # Module for loading and querying the simulation database
│   ├── env/                         # Navisim environment setup and Gym-compatible interface
│   ├── motion/                      # Motion model and trajectory simulator
│   ├── rendering/                   # Camera projection and Gaussian renderer wrapper
│   ├── space/                       # Spatial data structures (poses, transforms, coordinates)
│   ├── utils/                       # Helper functions and utilities used across modules
│   └── enums/                       # Enum class definitions for standardizing constant values
├── configs/                         # Experiment and environment configuration files
├── results/                         # Output logs, CSVs, and visual artifacts
├── setup.py                        # Package configuration and C++ extension build setup
└── README.md                       # Project documentation
```

---

## 🚨 Troubleshooting

### Common Build Issues

**Missing C++ compiler:**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install

# Windows
# Install Visual Studio Build Tools
```

**CUDA not found (optional):**
- Ensure CUDA toolkit is installed if you want GPU acceleration
- The package will fall back to CPU-only mode if CUDA is unavailable

**Import errors after building:**
- Try: `python -c "import sys; print(sys.path)"`
- Ensure you're in the correct conda environment
- Rebuild with: `pip install -e . --force-reinstall`

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

For questions or contributions, please open an issue or contact [email@example.com](mailto:emaill@example.com).
