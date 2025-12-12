#!/bin/bash
# docker_run.sh - Run NaviSim IsaacLab environment in Docker container

# Configuration
ISAAC_SIM_IMAGE="nvcr.io/nvidia/isaac-sim:4.2.0"  # Adjust version as needed
WORKSPACE_DIR="/home/jiwon-hae/Dev/navisim"
CONTAINER_NAME="navisim_isaaclab"

# Docker run command with GPU access and workspace mount
docker run --rm -it \
    --gpus all \
    --name ${CONTAINER_NAME} \
    --network host \
    -v ${WORKSPACE_DIR}:/workspace/navisim \
    -w /workspace/navisim/navisim_isaacsim \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    ${ISAAC_SIM_IMAGE} \
    bash -c "
        echo '[NaviSim] Setting up environment...'

        # Install Isaac Lab if not already installed
        if [ ! -d /workspace/isaaclab ]; then
            echo '[NaviSim] Isaac Lab not found in container'
            exit 1
        fi

        # Add Isaac Lab to Python path
        export PYTHONPATH=/workspace/isaaclab/source:\${PYTHONPATH}

        # Run the navigation environment
        python3 scripts/run.py \$@
    " -- "$@"
