#!/bin/bash
# run_isaac_lab.sh - Run NaviSim IsaacLab scripts in Docker

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Docker configuration
ISAAC_LAB_IMAGE="nvcr.io/nvidia/isaac-lab:2.3.0"
CONTAINER_NAME="isaac-lab-navisim"

# Default command to run inside container
# Isaac Lab container has python available via /isaac-sim/python.sh
if [ $# -eq 0 ]; then
    COMMAND="/isaac-sim/python.sh scripts/run_with_jetbot_camera.py --num_envs 1"
else
    # Prepend /isaac-sim/python.sh to provided command if it starts with a script
    if [[ "$1" == *.py ]]; then
        COMMAND="/isaac-sim/python.sh $@"
    else
        COMMAND="$@"
    fi
fi

echo "=========================================="
echo "Running NaviSim with Isaac Lab"
echo "=========================================="
echo "Container: ${CONTAINER_NAME}"
echo "Image: ${ISAAC_LAB_IMAGE}"
echo "Command: ${COMMAND}"
echo "=========================================="
echo ""

# Run Docker container
# X11 forwarding for visualization
xhost +local:docker > /dev/null 2>&1

docker run --name ${CONTAINER_NAME} --entrypoint bash --gpus all \
   -e "ACCEPT_EULA=Y" \
   -e "PRIVACY_CONSENT=Y" \
   -e "DISPLAY=${DISPLAY}" \
   -e "QT_X11_NO_MITSHM=1" \
   --rm \
   --network=host \
   -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   -v "${SCRIPT_DIR}":/workspace/user:rw \
   --workdir /workspace/user \
   ${ISAAC_LAB_IMAGE} \
   -c "${COMMAND}"
