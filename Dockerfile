FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Install CuPy
RUN pip3 install cupy-cuda117

# Install the required libraries
RUN pip3 install opencv-python numpy cupy cusignal cupyx cupyx.scipy matplotlib

# Install additional dependencies
RUN apt-get install -y libgl1-mesa-glx

# Install other required libraries
RUN pip3 install scipy

# Install the remaining libraries
RUN pip3 install matplotlib opencv-python-headless

# Install the remaining dependencies
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# # Optional: Set up a working directory
# WORKDIR /workspace

# # Optional: Copy your project files into the Docker container
# COPY . /workspace


# Set the working directory in the container
WORKDIR /app

# Copy the rest of the repository files to the working directory
COPY . .

# Specify the command to run when the container starts
CMD [ "python", "app.py" ]