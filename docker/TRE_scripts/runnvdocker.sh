#!/bin/bash

# Check if exactly two arguments are passed
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 [-d|-it] <docker-image-id> <container-command>"
    echo "use -d for detached mode, and -it for interactive mode"
    exit 1
fi

RUN_MODE=$1
UCLID=$(whoami)
IMAGE=$2
shift 2
CONTAINER_CMD=$@

# Run the container only if the user's home directory exists
if [ -e "/home/$UCLID" ]; then
    mkdir -p /home/$UCLID/data
    if [ "$RUN_MODE" = "-it" ]; then
        podman run -it --rm --privileged --gpus all -v /home/$UCLID/data:/workspace/data:z -name dev-container $IMAGE $CONTAINER_CMD
    else
        podman run -d --rm --privileged --gpus all -v /home/$UCLID/data:/workspace/data:z -name dev-container $IMAGE $CONTAINER_CMD
    fi
else
    echo "There is no home directory for user $UCLID"
    echo "Please check the UCL id you used"
    exit 1
fi
