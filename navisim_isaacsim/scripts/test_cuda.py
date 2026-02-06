#!/usr/bin/env python3
"""Test CUDA and PyTorch configuration."""

import os
import sys

print("="*60)
print("CUDA and PyTorch Configuration Test")
print("="*60)

# Check environment variables
print("\n1. Environment Variables:")
print(f"   TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST', 'Not set')}")
print(f"   CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"   PYTORCH_JIT: {os.environ.get('PYTORCH_JIT', 'Not set (default: 1)')}")

# Check PyTorch
try:
    import torch
    print("\n2. PyTorch:")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"   GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"      Name: {torch.cuda.get_device_name(i)}")
            cap = torch.cuda.get_device_capability(i)
            print(f"      Compute capability: {cap[0]}.{cap[1]}")
            props = torch.cuda.get_device_properties(i)
            print(f"      Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"      Multi-processor count: {props.multi_processor_count}")
    else:
        print("   ❌ CUDA not available!")
        sys.exit(1)

except ImportError as e:
    print(f"\n   ❌ PyTorch not found: {e}")
    sys.exit(1)

# Test basic CUDA operation
print("\n3. Testing CUDA operations:")
try:
    x = torch.randn(10, 10).cuda()
    y = torch.randn(10, 10).cuda()
    z = torch.matmul(x, y)
    print(f"   ✓ Basic CUDA operations work")
    print(f"   Result shape: {z.shape}")
except Exception as e:
    print(f"   ❌ CUDA operation failed: {e}")
    sys.exit(1)

# Test JIT compilation
print("\n4. Testing JIT compilation:")
try:
    @torch.jit.script
    def test_jit(x, y):
        return x + y

    result = test_jit(x, y)
    print(f"   ✓ JIT compilation works")
except Exception as e:
    print(f"   ⚠️  JIT compilation failed: {e}")
    print(f"   This is the issue causing your error!")
    print(f"\n   Fix: Set TORCH_CUDA_ARCH_LIST or disable JIT")

# Test normalize operation (the one failing in your error)
print("\n5. Testing normalize operation (from error traceback):")
try:
    from isaaclab.utils import math_utils
    gravity = torch.tensor([0.0, 0.0, -9.81], device='cuda')
    normalized = math_utils.normalize(gravity.unsqueeze(0)).squeeze(0)
    print(f"   ✓ Normalize operation works: {normalized}")
except Exception as e:
    print(f"   ❌ Normalize operation failed: {e}")
    print(f"   This is exactly your error!")

print("\n" + "="*60)
print("Test complete!")
print("="*60)
