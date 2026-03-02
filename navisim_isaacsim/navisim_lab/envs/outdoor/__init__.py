from .outdoor_scene_cfg import OutdoorSceneCfg
from .outdoor_env_cfg import OutdoorEnvCfg

# Camera variants are intentionally NOT imported here.
# Importing them sets rtx_sensors=True (via isaaclab.sensors.camera) and causes
# headless runs without --enable_cameras to hang.  Import them explicitly only
# when you need camera capture:
#   from navisim_lab.envs.outdoor.outdoor_scene_with_camera_cfg import OutdoorSceneWithCameraCfg
#   from navisim_lab.envs.outdoor.outdoor_env_with_camera_cfg import OutdoorEnvWithCameraCfg
