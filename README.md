# Grads, Rands & Nets

Docker

```
docker build -f docker/base/Dockerfile -t gran:latest .
docker build -f docker/no_gpu/Dockerfile -t gran:latest .
docker build -f docker/no_mpi/Dockerfile -t gran:latest .
```
```
#                       MPI in Docker JNotebook                                                        RAM access
docker run --name granc --privileged -p 8888:8888 --gpus=all -it -v /home/max/Dropbox/gran:/gran -v /dev/shm/:/dev/shm/  gran:latest
```
```
cd gran/
pip install -e .
python3 -m gran.rands.ga -n 4 -e envs/multistep/score/control.py -b bots/network/static/rnn/control.py -g 100 -p 16 -a '{"task" : "cart_pole", "transfer" : "no"}'
python3 -m gran.rands.evaluate -n 4 -p data/states/envs.multistep.score.control/seeding.reg~steps.0~task.cart_pole~transfer.no~trials.1/bots.netted.static.rnn.control/16/ -s 0
jupyter notebook --allow-root --ip 0.0.0.0
```

