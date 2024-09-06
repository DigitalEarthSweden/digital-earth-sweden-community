FROM ubuntu:22.04 as eo_training_base

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

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
    echo 'Running $(conda --version)' && \
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

# Set the PYTHONPATH to include the project
ENV PYTHONPATH="/proj/tutorials:${PYTHONPATH}"

# Set the working directory to /proj
WORKDIR /proj

# Expose the necessary ports
EXPOSE 8888

# Set the entry point to automatically start Jupyter Lab
ENTRYPOINT ["bash", "-c", "source activate openeo-training && jupyter lab --port=8888 --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password='' --notebook-dir=/proj --no-browser"]
