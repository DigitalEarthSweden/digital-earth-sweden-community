# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.10.2@sha256:94a23af2d50e97b87b522d3cea24aaf8a1faedec1344c952767434f69585cbf9 AS uv

FROM ghcr.io/osgeo/gdal:ubuntu-full-3.12.2@sha256:35cf3b42728f568b911a120d8644b5e5fb7d277b8d3ed07e4b66c0af19c957af AS internal_base

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.12 \
    UV_PROJECT_ENVIRONMENT=/app

FROM internal_base AS builder

WORKDIR /build

COPY --link --from=uv /uv /uvx /usr/local/bin/

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      gcc \
      g++ \
      libc6-dev \
      linux-libc-dev \
      libgeos-dev \
      libhdf5-dev \
      libnetcdf-dev \
      libproj-dev \
      libudunits2-dev \
      python3-dev

COPY --link pyproject.toml uv.lock /build/

# Use a separate cache volume for uv on openeo-training, so it is
# not inseparable from pip/poetry/npm/etc. cache stored in /root/.cache.
RUN --mount=type=cache,id=openeo-training-uv-cache,target=/root/.cache \
    uv sync --locked --no-dev --no-install-project \
      --no-binary-package fiona \
      --no-binary-package rasterio \
      --no-binary-package shapely


FROM internal_base AS prod

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      gosu \
      python3 \
      sudo \
      tree \
      wget

RUN usermod -aG sudo ubuntu && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Docker 28.x requires numeric uid/gid with --link when using
# a non-default builder like the CI action does in this repository.
COPY --from=builder --link --chown=1000:1000 /app /app

USER ubuntu
WORKDIR /home/ubuntu

# Copy the tutorials directory to /proj
COPY --chown=ubuntu:ubuntu tutorials /proj/tutorials

# Copy the tests directory to /test
COPY --chown=ubuntu:ubuntu tests /test

# Copy the input directory to /input
COPY --chown=ubuntu:ubuntu input /proj/input
# Set the PYTHONPATH to include the project
ENV PYTHONPATH="/proj/tutorials:/app" \
    PATH="/app/bin:$PATH"

# Set the working directory to /proj
WORKDIR /proj

# Expose the necessary ports
EXPOSE 8888

# Set the entry point to automatically start Jupyter Lab
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser","--allow-root","--IdentityProvider.token=''"]
