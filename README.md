# Digital-Earth-Sweden Community

This is where you can:
- Take an extensive tutorial of our system
- File bug reports
- Make wishes for and discuss new features
- Ask and answer questions

The community repo is shared and monitored by developers, stakeholders and regular users. It is the responsibility of us all to look into this repository and help each other out.

# Tutorials
This guide will walk you through various ways to run the Jupyter notebooks included in this repository, whether you just want to explore the content or dive deep into working with the files.

## Running a Notebook Server Without Cloning the Repo

If you want to quickly check out the tutorials without saving any changes or downloading the files locally, you can run the notebook server directly from the Docker image. This is a great option if you’re just exploring and don’t need to save your work. If you do not have Docker installed or know how to use it check out [https://docs.docker.com/desktop/](https://docs.docker.com/desktop/).

### How to Run

1. Open your terminal or command prompt.
2. Run the following Docker commands:

   `docker pull ghcr.io/digitalearthsweden/tutorials:latest`

   `docker run --rm -it -p 8888:8888 ghcr.io/digitalearthsweden/tutorials:latest`

3. Open Firefox (or your preferred browser) and navigate to `http://127.0.0.1:8888` to access the Jupyter Lab interface.

   If using Firefox, you can also start the browser with the URL directly from the command line:

   `firefox http://127.0.0.1:8888`

   This will allow you to explore the tutorials without needing to clone the repository or worry about saving files.

### Note
- **Files are not saved**: Since you’re running the notebooks in a Docker container without mounting any local directories, any changes you make will not persist after the container is stopped.

## If You Want to Work with the Files

If you want to make changes to the notebooks and save your work, you’ll need to clone the repository and run the notebooks in a way that allows you to persist files.

### How to Clone and Run

1. **Clone the Repository**:

   First, clone the repository to your local machine:

   **HTTPS**: `git clone https://github.com/DigitalEarthSweden/digital-earth-sweden-community.git`
   
   **SSH**: `git clone git@github.com:DigitalEarthSweden/digital-earth-sweden-community.git`

   and then `cd digital-earth-sweden-community`

2. **Run the following docker commands**:

   `docker pull ghcr.io/digitalearthsweden/tutorials:latest`

   `docker run --rm -it -p 8888:8888 --mount type=bind,source=./tutorials,target=/proj ghcr.io/digitalearthsweden/tutorials:latest`

   These commands will start the Docker container, mount the tutorials directory, and expose the Jupyter Lab server at `http://127.0.0.1:8888`.

- **Mounting**: Mounting your local directory to the container allows the container to read and write files from your local machine. This ensures that any changes you make in Jupyter Lab, including new files and folders, will be saved in the tutorials folder.

## Setting Up a Local Environment

If you prefer to work in your local environment without Docker, you can create a Python virtual environment from the `pyproject.toml` in the repository. We recommend [uv](https://docs.astral.sh/uv/), since it installs exactly the same package versions as the Docker image (from `uv.lock`). Please do not use Conda for these tutorials; mixing package managers easily leads to versioning problems.

### Option 1: uv (recommended)

1. **Install uv**: Follow the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

2. **Create the Environment**:

   Navigate to the directory where you cloned the repository and run:

   `uv sync`

   This creates a virtual environment in `.venv` with all the dependencies needed to run the notebooks. If the Python version pinned in `.python-version` is not installed, uv will download it for you.

3. **Run Jupyter Lab**:

   `uv run jupyter lab`

   This will launch Jupyter Lab in your default web browser, allowing you to work with the tutorials in a local environment.

### Option 2: pip

1. **Create and Activate a Virtual Environment** (requires Python 3.12 or newer):

   `python -m venv .venv`

   `source .venv/bin/activate` (on Windows: `.venv\Scripts\activate`)

2. **Install the Dependencies**:

   `pip install .`

   Note that pip does not use the lock file, so it installs the newest package versions allowed by `pyproject.toml`. These may differ from the versions in the Docker image.

3. **Run Jupyter Lab**:

   `jupyter lab`

### Running the Tests

The notebooks are tested by executing them against the Digital Earth Sweden platform:

`uv run pytest tests/`

### Requirements
- **uv** (recommended), or **Python 3.12 or newer** with pip.

# Additional Resources
If you are new to Digital Earth Sweden, the following resources may be helpful:
- https://maps.digitalearth.se
- https://explorer.digitalearth.se
- https://editor.openeo.org/?server=https%3A%2F%2Fopeneo.digitalearth.se


**NOTE!** Since our team is very small, we will take turns to monitor this forum. Typically this will be assigned a slot on Mondays. Please avoid using Teams and personal messages to the team.
