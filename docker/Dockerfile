# docker build --network host . -t dev:latest

FROM nvcr.io/nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Prevents `apt-get install opencv-python` from prompting geographical details
ENV DEBIAN_FRONTEND=noninteractive 

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

RUN mpi4py==3.1.3 \
    torch==1.13.0 --extra-index-url https://download.pytorch.org/whl/cu117 \
    torchvision==0.14.0 --extra-index-url https://download.pytorch.org/whl/cu117 \
    opencv-python==4.6.0.66 \
    gym[all]==0.26.2