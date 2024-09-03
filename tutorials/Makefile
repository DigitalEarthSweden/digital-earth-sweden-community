# Makefile

# Define the image name and tag
IMAGE_NAME = eo-training
IMAGE_TAG = latest


# Define the source directory to mount (for start-mount-dev)
SOURCE_DIR = $(shell pwd)

list-targets:
	@grep -E '^[^[:space:]]+:.*' Makefile | grep -v '=' | cut -d ':' -f 1


# Build the Docker image
build: 
	docker build --progress=plain -t $(IMAGE_NAME):$(IMAGE_TAG) .

# Start the Docker container without mounting the source directory
start-dev: build
	docker run -it --rm $(IMAGE_NAME):$(IMAGE_TAG)

# Start the Docker container and mount the source directory
start-mount-dev: build
	docker run -it --rm -v $(SOURCE_DIR):/proj $(IMAGE_NAME):$(IMAGE_TAG) /bin/bash 

# Run pytest on the tests directory inside the container
test-notebooks-docker: build
	docker run -it --rm $(IMAGE_NAME):$(IMAGE_TAG) bash -ic "pytest -v tests"

start-notebook: build
	@echo "==============================================================================  "
	@echo "             Start a notebook in a  container mounted to local drive            "
	@echo "             NOTE that the container SHARES your files on your HDD now          "
	@echo "             under the folder ./proj "
	@echo "=============================================================================== "
	docker run --rm -it -p 8888:8888 --name eo-training --mount type=bind,source=$(shell pwd),target=/proj $(IMAGE_NAME):$(IMAGE_TAG)