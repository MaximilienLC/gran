FROM nvcr.io/nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Prevents `apt-get install opencv-python` from prompting geographical details
ENV DEBIAN_FRONTEND=noninteractive 
# Prevents Python from creating __pycache__ and .pyc files in project folder
ENV PYTHONPYCACHEPREFIX=/.cache/python/

RUN apt-get update && apt-get install -y \
    # Python
    python3-dev \
    python3-pip \
    # OpenMPI
    libopenmpi-dev \
    # Parallel HDF5
    libhdf5-openmpi-dev \
    python3-h5py-mpi \
    # OpenCV requires system packages, best to install through apt-get 
    python3-opencv \
    # Box2D Requirement
    swig \
    # MuJoCo Requirements
    wget \
    libosmesa6-dev \
    patchelf \
    # Github Python Package
    git \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/Downloads/ \
    && mkdir -p ~/.mujoco/ && tar -zxf ~/Downloads/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/ \
    && rm ~/Downloads/mujoco210-linux-x86_64.tar.gz \
    && echo "\n# MuJoCo\nMUJOCO_PATH=~/.mujoco/mujoco210/bin\nexport LD_LIBRARY_PATH=\$MUJOCO_PATH:\$LD_LIBRARY_PATH" >> ~/.bashrc

ADD reqs /pip_reqs/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --pre torch==2.0.0.dev20230120 --index-url https://download.pytorch.org/whl/nightly/cu117 \
    && pip install --no-cache-dir --pre torchvision==0.15.0.dev20230120 --index-url https://download.pytorch.org/whl/nightly/cu117 \
    && pip install --no-cache-dir -r /pip_reqs/sweep_reqs.txt -r /pip_reqs/run_reqs.txt
