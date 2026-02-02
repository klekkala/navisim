#!/usr/bin/env python3
"""Simple import test to verify code structure (no torch/Isaac Sim required)."""

import sys
from pathlib import Path
import ast

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_file_syntax(file_path: Path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, "r") as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)


def main():
    """Check syntax of all refactored files."""
    print("=" * 80)
    print("NaviSim Agent Framework - Syntax Check")
    print("=" * 80)

    files_to_check = [
        "navisim_lab/agents/base_agent.py",
        "navisim_lab/agents/__init__.py",
        "navisim_lab/agents/rsl_rl/rsl_rl_agent.py",
        "navisim_lab/agents/rsl_rl/__init__.py",
        "navisim_lab/envs/base/base_nav_env.py",
        "navisim_lab/envs/base/__init__.py",
        "navisim_lab/envs/warehouse/warehouse_env.py",
        "scripts/train.py",
        "scripts/play.py",
    ]

    all_passed = True

    for file_path_str in files_to_check:
        file_path = project_root / file_path_str
        if not file_path.exists():
            print(f"✗ {file_path_str}: FILE NOT FOUND")
            all_passed = False
            continue

        valid, error = check_file_syntax(file_path)
        if valid:
            print(f"✓ {file_path_str}")
        else:
            print(f"✗ {file_path_str}: {error}")
            all_passed = False

    print("=" * 80)

    # Check directory structure
    print("\nDirectory Structure:")
    dirs_to_check = [
        "navisim_lab/agents",
        "navisim_lab/agents/rsl_rl",
        "navisim_lab/envs/base",
        "scripts",
    ]

    for dir_path_str in dirs_to_check:
        dir_path = project_root / dir_path_str
        if dir_path.exists():
            print(f"✓ {dir_path_str}/")
        else:
            print(f"✗ {dir_path_str}/: DIRECTORY NOT FOUND")
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("✓ All syntax checks passed!")
        print("\nNext steps:")
        print("1. Run smoke_test.py to verify environment still works")
        print("2. Use new unified scripts:")
        print("   python scripts/train.py --task Navisim-Warehouse-Jetbot-v0 \\")
        print("       --agent rsl_rl --config navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml")
        return 0
    else:
        print("✗ Some checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
