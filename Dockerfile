FROM python:3.13-slim

# Install necessary packages
RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y --no-install-recommends tree gosu sudo wget && \
    apt-get install -y --no-install-recommends ca-certificates && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Create user, add to sudo group, configure sudoers.
RUN adduser --disabled-password --gecos '' ubuntu && \
    usermod -aG sudo ubuntu && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ubuntu
WORKDIR /home/ubuntu

ENV PATH="/home/ubuntu/miniconda3/bin:${PATH}"
ARG PATH="/home/ubuntu/miniconda3/bin:${PATH}"

ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh; \
    else \    
      wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; \
    fi 

RUN bash miniconda.sh -b && \
    rm -f miniconda.sh && \
    /home/ubuntu/miniconda3/bin/conda init && \
    . ~/.bashrc && \
    conda update conda -y && \
    conda install -y -c conda-forge mamba

# Create the Conda environment
COPY --chown=ubuntu:ubuntu tutorials/environment.yml /home/ubuntu/environment.yml
RUN conda env create -f /home/ubuntu/environment.yml && \
    conda clean -afy

# Copy the tutorials directory to /proj
COPY --chown=ubuntu:ubuntu tutorials /proj/tutorials

# Copy the tests directory to /test
COPY --chown=ubuntu:ubuntu tests /test

# Copy the input directory to /input
COPY --chown=ubuntu:ubuntu input /proj/input
# Set the PYTHONPATH to include the project
ENV PYTHONPATH="/proj/tutorials:/app"

# Set the working directory to /proj
WORKDIR /proj

# Expose the necessary ports
EXPOSE 8888

# Set the entry point to automatically start Jupyter Lab
CMD ["bash", "-c", "source activate openeo-training && jupyter lab --port=8888 --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password='' --notebook-dir=/proj --no-browser"]
