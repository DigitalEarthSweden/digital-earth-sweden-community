# Digital-Earth-Sweden Community

This is where you can:
- Take an extensive tutorial of our system
- File bug reports
- Make wishes for and discuss new features
- Ask and answer questions

The community repo is shared and monitored by developers, stakeholders and regular users. It is the resposibility for us all to look into this repository and help each other out. 
# Tutorials
 This guide will walk you through various ways to run the Jupyter notebooks included in this repository, whether you just want to explore the content or dive deep into working with the files.

## Running a Notebook Server Without Cloning the Repo

If you want to quickly check out the tutorials without saving any changes or downloading the files locally, you can run the notebook server directly from the Docker image. This is a great option if you’re just exploring and don’t need to save your work.

### How to Run

1. Open your terminal or command prompt.
2. Run the following Docker command:

   `docker run --rm -it -p 8888:8888 harbor.main.rise-ck8s.com/des-public/tutorials:latest`

4. Open Firefox (or your preferred browser) and navigate to the URL provided in the terminal output to access the Jupyter Lab interface.

   If using Firefox, you can also start the browser with the URL directly from the command line (replace `your_token_here` with the actual token):

   `firefox http://127.0.0.1:8888

   This will allow you to explore the tutorials without needing to clone the repository or worry about saving files.

### Note
- **Files are not saved**: Since you’re running the notebooks in a Docker container without mounting any local directories, any changes you make will not persist after the container is stopped.

## If You Want to Work with the Files

If you want to make changes to the notebooks and save your work, you’ll need to clone the repository and run the notebooks in a way that allows you to persist files.

### How to Clone and Run

1. **Clone the Repository**:

   First, clone the repository to your local machine:

   `git clone https://github.com/your_username/your_repository.git`

   `cd your_repository`

2. **Use the Start Scripts**:

   After cloning, you can use the provided start scripts to run the notebook server with your local directory mounted inside the container. This means that any changes you make to the notebooks will be saved locally.

   - On **Windows**:

     Run the `start_notebook.BAT` script:

     `start_notebook.BAT`

   - On **macOS/Linux**:

     Run the `start_notebook.sh` script:

     `./start_notebook.sh`

   These scripts will start the Docker container, mount the current directory, and expose the Jupyter Lab server at `http://127.0.0.1:8888`. Your files will be saved locally in the directory you cloned.

### What It Means to Mount Locally
- **Mounting**: Mounting your local directory to the container allows the container to read and write files from your local machine. This ensures that any changes you make in Jupyter Lab are reflected in your local files.

## Setting Up a Local Environment

If you prefer to work in your local environment without Docker, you can set up the Conda environment defined in the repository.

### Steps to Set Up

1. **Install Conda**: If you don’t already have Conda installed, you can download and install it from the [official site](https://docs.conda.io/en/latest/miniconda.html).

2. **Create the Environment**:

   Navigate to the directory where you cloned the repository and create the Conda environment:

   `conda env create -f environment.yml`

   This command will create a Conda environment with all the dependencies needed to run the notebooks.

3. **Activate the Environment**:

   Activate the environment:

   `conda activate openeo-training`

4. **Run Jupyter Lab**:

   Start Jupyter Lab:

   `jupyter lab`

   This will launch Jupyter Lab in your default web browser, allowing you to work with the tutorials in a local environment.

### Requirements
- **Conda**: Make sure you have Conda installed to manage the environment.

## Building the Notebook Server

If you need to build the Docker image and run the notebooks yourself, you can use the provided Makefile.

### How to Build and Run

1. **Install Requirements**:

   Ensure that you have Docker and Make installed on your system.

2. **Build the Notebook Server**:

   Use the following command to build the Docker image and start the notebook server:

   `make start-notebook`

   This command will:
   - Build the Docker image using the `Dockerfile`.
   - Mount your local directory into the container.
   - Start the Jupyter Lab server on `http://127.0.0.1:8888`.


# Additional Resources 
If you are new to Digital Earth Sweden, please check  
Also the following resources may be helpful:
- https://maps.digitalearth.se
- https://explorer.digitalearth.se
- https://editor.openeo.org/?server=https%3A%2F%2Fopeneo.digitalearth.se


**NOTE!** Since our team is very small, we will take turn to monitor this forum. Typically this will be assigned a slot on mondays. Please avoid using Teams and personal messages to the team. 
