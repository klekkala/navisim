
from pathlib import Path

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

WAREHOUSE_USD = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd"
JETBOT_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"

# Custom 3DGS-converted environment (no physics — uses ground plane for collision)
CUSTOM_ENV_USD = str(Path(__file__).parents[2] / "assets" / "new_point_cloud.usdz")