#!/bin/bash

# Variables
IMAGE_NAME="harbor.main.rise-ck8s.com/des-public/tutorials:latest"
NOTEBOOK_PORT=8888
HOST_DIR=$(pwd)  # The current directory
CONTAINER_DIR="/proj"  # Directory inside the container

# Run the Docker container
docker pull $IMAGE_NAME
docker run --rm -it \
    -p $NOTEBOOK_PORT:8888 \
    --name eo-training-notebook \
    --mount type=bind,source=$HOST_DIR,target=$CONTAINER_DIR \
    $IMAGE_NAME

# Explanation:
# - `--rm`: Automatically remove the container when it exits.
# - `-it`: Interactive terminal.
# - `-p 8888:8888`: Map port 8888 on the host to port 8888 in the container.
# - `--mount`: Mount the current directory into the container at /proj.
# - `$IMAGE_NAME`: The Docker image to run, which now automatically starts Jupyter Lab.
