FROM nvcr.io/nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip

COPY requirements.txt /dl/requirements.txt
RUN pip install -r /dl/requirements.txt

# docker run --gpus=all -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -v $PWD:/home/womo/ --volume="/run/user/1000/gdm/Xauthority:/root/.Xauthority:rw" --env="DISPLAY" -w /home/ womo:1.0 bash