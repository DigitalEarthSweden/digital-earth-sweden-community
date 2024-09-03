@echo off

REM Variables
set IMAGE_NAME=harbor.main.rise-ck8s.com/des-public/tutorials:latest
set NOTEBOOK_PORT=8888
set HOST_DIR=%cd%
set CONTAINER_DIR=/proj

REM Run the Docker container
docker run --rm -it ^
    -p %NOTEBOOK_PORT%:8888 ^
    --name eo-training-notebook ^
    --mount type=bind,source=%HOST_DIR%,target=%CONTAINER_DIR% ^
    %IMAGE_NAME%

REM Explanation:
REM - `--rm`: Automatically remove the container when it exits.
REM - `-it`: Interactive terminal.
REM - `-p 8888:8888`: Map port 8888 on the host to port 8888 in the container.
REM - `--mount`: Mount the current directory into the container at /proj.
REM - `%IMAGE_NAME%`: The Docker image to run, which now automatically starts Jupyter Lab.
