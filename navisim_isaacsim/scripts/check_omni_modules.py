"""Check what omni modules are available."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("\n" + "="*80)
print("Checking available omni modules...")
print("="*80)

# Try importing various omni modules
modules_to_check = [
    "omni.replicator.core",
    "omni.isaac.core.utils.render",
    "omni.kit.viewport.utility",
    "omni.usd",
    "pxr",
]

available = []
unavailable = []

for module_name in modules_to_check:
    try:
        __import__(module_name)
        available.append(module_name)
        print(f"✓ {module_name} - AVAILABLE")
    except ImportError as e:
        unavailable.append((module_name, str(e)))
        print(f"✗ {module_name} - NOT AVAILABLE: {e}")

print("\n" + "="*80)
print(f"Summary: {len(available)}/{len(modules_to_check)} modules available")
print("="*80)

if "omni.replicator.core" in [m for m in available]:
    print("\n✓ omni.replicator.core is available - camera capture should work!")
else:
    print("\n✗ omni.replicator.core is NOT available")
    print("   This Isaac Lab environment may not have Replicator enabled.")
    print("   Camera capture will not work with the current approach.")

simulation_app.close()
