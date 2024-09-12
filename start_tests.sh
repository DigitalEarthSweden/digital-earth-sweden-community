#!/bin/bash

# Set environment variables
DES_DOCKER_IMAGE=${DES_DOCKER_IMAGE:-"harbor.main.rise-ck8s.com/des-public/tutorials:latest"}
CONTAINER_NAME="des_notebooks"
JUPYTER_PORT=9999  # Changed to 9999 to avoid conflicts

# Clean up any previous runs
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Start the Docker container with Jupyter in the background
docker run -d --rm --name $CONTAINER_NAME -p $JUPYTER_PORT:9999 $DES_DOCKER_IMAGE

# Wait for the Jupyter server to start (you might need to adjust this time)
echo "Waiting for the Jupyter server to start..."
sleep 10

# Check if the container is running
if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "Failed to start the Docker container!"
    exit 1
fi

# Run pytest inside the container and capture the result
docker exec $CONTAINER_NAME bash -c "source activate des-community && pytest --disable-warnings /test/test_notebooks.py"
TEST_RESULT=$?

# Stop and remove the container
docker stop $CONTAINER_NAME

# Exit with the test result code
exit $TEST_RESULT
