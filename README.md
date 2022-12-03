# Grads, Rands & Nets

Docker

docker build . -t dev:latest
docker run --name devc --gpus=all -it -v /home/max/Dropbox/gran:/gran -v /dev/shm/:/dev/shm/ dev:latest
