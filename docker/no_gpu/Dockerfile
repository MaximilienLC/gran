FROM ubuntu:20.04

# Prevents `apt-get install opencv-python` from prompting geographical details
ENV DEBIAN_FRONTEND=noninteractive 
# Prevents Python from creating __pycache__ and .pyc files in project folder
ENV PYTHONPYCACHEPREFIX=/.cache/python/

RUN apt-get update && apt-get install -y \
    wget \
    python3-pip \
    libopenmpi-dev \
    # OpenCV requires system packages, best to install through apt-get 
    python3-opencv \
    # Box2D Requirement
    swig

RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/Downloads/
RUN mkdir -p ~/.mujoco/ && tar -zxf ~/Downloads/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/

RUN pip install -r reqs/torch_cpu.txt -r reqs/mpi4py.txt -r reqs/other_reqs.txt