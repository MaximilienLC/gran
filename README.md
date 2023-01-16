# Grads, Rands & Nets

Docker

```
docker build -f docker/Dockerfile -t gran:latest .
```
```
#                       MPI in Docker JNotebook                                                        RAM access
docker run --name granc --privileged -p 8888:8888 --gpus=all -it -v /home/max/Dropbox/gran:/gran -w /gran -v /dev/shm/:/dev/shm/  gran:latest
```
```
#pip install -e .
#python3 -m gran.rands.ga
#python3 -m gran.rands.evaluate
python3 gran/rands/ga.py
python3 gran/rands/evaluate.py
jupyter notebook --allow-root --ip 0.0.0.0
```
```
# On Compute Canada
module load python/3.10.2
python3 -m venv /scratch/mleclei/venv
. /scratch/mleclei/venv/bin/activate
pip install -r hydra_reqs.txt
sed -i "s/shlex.quote(sys.executable)/\"apptainer exec --nv --bind \/scratch\/mleclei\/gran:\/gran --pwd \/gran \/scratch\/mleclei\/gran.sif python3\"/" ../venv/lib/python3.10/site-packages/submitit/slurm/slurm.py
```