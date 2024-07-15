import sys
from pathlib import Path

# Calculate the path to the module_to_copy directory
module_path = Path(__file__).resolve().parent / 'gaussian'

# Add the module_to_copy directory to sys.path
if str(module_path) not in sys.path:
    sys.path.append(str(module_path))