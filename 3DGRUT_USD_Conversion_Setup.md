# 3DGRUT USD Conversion -- Pre-Execution Setup Guide

**Environment Context**\
This setup was validated on: - Server: `ilab` - Machine: `iGPU10`

This environment is **shared**.\
Do **NOT** reinstall CUDA, modify system drivers, or change Conda
packages.

This document explains what must be set **before running PLY → USDZ
conversion**.

------------------------------------------------------------------------

## Conversion Command

``` bash
python -m threedgrut.export.scripts.ply_to_usd     input.ply     --output_file output.usdz
```

------------------------------------------------------------------------

# Required Setup (Run Before Conversion)

## 1️⃣ Activate the Environment

``` bash
conda activate 3dgrut
```

------------------------------------------------------------------------

## 2️⃣ Remove Runtime Overrides

Slang and Torch rely on internal runtime resolution.

``` bash
unset LD_LIBRARY_PATH
```

Do NOT manually set `LD_LIBRARY_PATH`.

------------------------------------------------------------------------

## 3️⃣ Expose CUDA Driver to Linker (Critical)

The system CUDA driver is located at:

    /lib/x86_64-linux-gnu/libcuda.so

The build process requires this path at **link time**.

Set:

``` bash
export LIBRARY_PATH=/lib/x86_64-linux-gnu
```

This allows the extension to link against `-lcuda`.

------------------------------------------------------------------------

## 4️⃣ Clear Torch Extension Cache (If Needed)

If a previous build failed:

``` bash
rm -rf ~/.cache/torch_extensions
```

------------------------------------------------------------------------

# Recommended Full Execution Block

``` bash
conda activate 3dgrut

unset LD_LIBRARY_PATH
export LIBRARY_PATH=/lib/x86_64-linux-gnu

rm -rf ~/.cache/torch_extensions

python -m threedgrut.export.scripts.ply_to_usd     input.ply     --output_file output.usdz
```

------------------------------------------------------------------------

# What NOT To Do

Do NOT install CUDA 12\
Do NOT symlink `libcudart.so`\
Do NOT modify system CUDA\
Do NOT permanently export `LD_LIBRARY_PATH`\
Do NOT reinstall `slangtorch`

------------------------------------------------------------------------

# Environment Expectations

  Component       Version
  --------------- ----------------------------
  Python          3.11
  PyTorch CUDA    11.8
  nvcc            11.8
  NVIDIA Driver   12.x (backward compatible)
  slangtorch      pip-installed

------------------------------------------------------------------------

# Why This Setup Works

-   PyTorch compiled for CUDA 11.8\
-   nvcc is 11.8\
-   NVIDIA driver (12.x) is backward compatible\
-   `LIBRARY_PATH` ensures `libcuda.so` is visible at link time\
-   `LD_LIBRARY_PATH` is left untouched to avoid Slang runtime conflicts

------------------------------------------------------------------------

# Optional Health Check

``` bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
ldconfig -p | grep libcuda
```

Expected: - Torch CUDA: `11.8` - nvcc: `11.8` - `libcuda.so` visible in
`/lib/x86_64-linux-gnu`

------------------------------------------------------------------------

# If Build Fails

Run:

``` bash
unset LD_LIBRARY_PATH
export LIBRARY_PATH=/lib/x86_64-linux-gnu
rm -rf ~/.cache/torch_extensions
```

Then retry conversion.

------------------------------------------------------------------------

# Status

This configuration is stable and reproducible on **ilab iGPU10**.

No further CUDA modifications are required.
