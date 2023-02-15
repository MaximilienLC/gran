# Gran: Grads, Rands & Neural Nets

## I. Setting up the repository

### Step 1. Download on your local machine
Define the library's path
```
LOCAL_GRAN_PATH=/home/stav/gran
LOCAL_GRAN_PATH=/home/max/Dropbox/gran
```
Clone the repository
```
git clone git@github.com:MaximilienLC/gran.git ${LOCAL_GRAN_PATH}
```
### Step 2. Build the Docker / Apptainer images locally

**(If you are in the `rrg-pbellec` Alliance Canada group you can skip to Step 3)**  
  
The following instructions define the full dependency installation steps on **Ubuntu** to ensure reproducibility across platforms, utilizing container technologies Docker & Apptainer (Singularity).  

Disclaimer 1: you can probably skip steps that you've performed for previous installations, however do note that that could potentially lead to installation failure.  

Disclaimer 2: you can probably install this repository in fewer steps if softwares like CUDA/cuDNN are already installed on your system, skipping the need for Docker & Apptainer.  
For the sake of simplicity & reproducibility, we however still recommend you follow this guide.

Disclaimer 3: multi-node MPI is not currently supported

#### i. Install Apptainer 
**(Skip this section if you have Docker privileges across machines)**
```
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt-get update
sudo apt-get install -y apptainer
```
Installation for other mediums: [[link](https://apptainer.org/docs/admin/main/installation.html)]

#### ii. Install and setup Docker
Install the deb Docker package (can also install from the official website: [[link](https://docs.docker.com/engine/install/)]).
```
sudo apt-get install -y docker.io
```
Give yourself docker permissions to not require sudo privileges for docker later on.
```
sudo groupadd docker
sudo usermod -aG docker ${USER}
```
Finally, log out and log back to apply changes.

#### iii. Install the NVIDIA Driver
1) "Show Applications"  
2) "Software & Updates"  
3) "Additional Drivers"  
4) Select: "Using NVIDIA driver metapackage from nvidia-driver-XXX (proprietery, tested)"  
  
Installation for other mediums: [[link](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)]

#### iv. Install the NVIDIA Container Toolkit
Setup the package repository and the GPG key.
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
Install the package and restart the docker engine.
```
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
Installation for other mediums: [[link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)]

#### v. Build the Docker image
```
cd ${LOCAL_GRAN_PATH}/
docker build -f docker/Dockerfile -t gran:latest .
```

#### v. Build the Apptainer image
**(Skip this section if you have Docker privileges across machines)**
```
cd ${LOCAL_GRAN_PATH}/docker/
docker save gran:latest -o image.tar
apptainer build image.sif docker-archive://image.tar
```

### Step 3. Set up the SLURM-based cluster
**(Skip this section if you do not need SLURM-based installation)**
Define path to the library.
```
CLUSTER_GRAN_PATH=/scratch/stavb/gran
CLUSTER_GRAN_PATH=/scratch/mleclei/Dropbox/gran
```
Clone & install Python packages.
```
git clone git@github.com:MaximilienLC/gran.git ${CLUSTER_GRAN_PATH}
module load python/3.10
python3 -m venv ${CLUSTER_GRAN_PATH}/venv
. ${CLUSTER_GRAN_PATH}/venv/bin/activate
pip install -r ${CLUSTER_GRAN_PATH}/reqs/hydra_reqs.txt
```
Hard-code fix to run Python through Apptainer using Submitit.
```
sed -i "s|shlex.quote(sys.executable)|\"apptainer exec --nv --bind ${CLUSTER_GRAN_PATH}:${CLUSTER_GRAN_PATH} --pwd ${CLUSTER_GRAN_PATH} --env PYTHONPATH=\${PYTHONPATH}:${CLUSTER_GRAN_PATH} ${CLUSTER_GRAN_PATH}\/docker\/image.sif python3\"|" ${CLUSTER_GRAN_PATH}/venv/lib/python3.10/site-packages/submitit/slurm/slurm.py
```
Copy the Apptainer image into your repository.
```
# Option 1 : copy over the images (rrg-pbellec + beluga cluster only)
cp /scratch/mleclei/Dropbox/gran/image.sif ${CLUSTER_GRAN_PATH}/docker/image.sif
cp /scratch/mleclei/Dropbox/gran/image.tar ${CLUSTER_GRAN_PATH}/docker/image.tar
  
# Option 2 : copy from your own machine (RUN THIS COMMAND ON YOUR LOCAL MACHINE)
scp ${LOCAL_GRAN_PATH}/docker/image.sif ${CLUSTER_USER}@${CLUSTER_ADDRESS}:${CLUSTER_GRAN_PATH}/docker/image.sif
```

### Step 4. Run

#### i. On your local machine
(rrg-pbellec group only) If you skipped Step 2, transfer the Docker image back to your local machine (RUN THIS COMMAND ON THE BELUGA CLUSTER)  
```
scp ${CLUSTER_USER}@${CLUSTER_ADDRESS}:${CLUSTER_GRAN_PATH}/docker/image.tar ${LOCAL_GRAN_PATH}/docker/.
docker load -i ${LOCAL_GRAN_PATH}/docker/image.tar
```
Execute the sample code
```
docker run --rm --privileged --gpus=all -e PYTHONPATH=${PYTHONPATH}:${LOCAL_GRAN_PATH} -v ${LOCAL_GRAN_PATH}:${LOCAL_GRAN_PATH} -w ${LOCAL_GRAN_PATH} -v /dev/shm/:/dev/shm/  gran:latest python3 gran/rand/main.py
```
Run notebooks
```
docker run --rm --privileged --gpus=all -p 8888:8888 -e PYTHONPATH=${PYTHONPATH}:${LOCAL_GRAN_PATH} -v ${LOCAL_GRAN_PATH}:${LOCAL_GRAN_PATH} -w ${LOCAL_GRAN_PATH} -v /dev/shm/:/dev/shm/  gran:latest jupyter notebook --allow-root --ip 0.0.0.0
```

#### iii. On a SLURM cluster
Activate the virtual environment
```
tmux
cd ${CLUSTER_GRAN_PATH}
. ${CLUSTER_GRAN_PATH}/venv/bin/activate
```
Execute the sample code
```
python3 gran/rand/main.py -m hydra/launcher=submitit_slurm +launcher=slurm
```
Run notebooks
```
module load apptainer/1.0
CLUSTER_GRAN_PATH=/scratch/mleclei/Dropbox/gran
salloc --account=rrg-pbellec --gres=gpu:1
apptainer exec --nv --bind ${CLUSTER_GRAN_PATH}:${CLUSTER_GRAN_PATH} --pwd ${CLUSTER_GRAN_PATH} --env PYTHONPATH=${PYTHONPATH}:${CLUSTER_GRAN_PATH} ${CLUSTER_GRAN_PATH}/docker/image.sif jupyter notebook --allow-root --ip $(hostname -f) --no-browser
sshuttle --dns -Nr mleclei@beluga.computecanada.ca
```

#### iii. On a new machine without Docker privileges
Define path to the library
```
NEW_GRAN_PATH=/home/mleclei/Dropbox/gran
```
Execute the sample code
```
apptainer exec --nv --bind ${NEW_GRAN_PATH}:${NEW_GRAN_PATH} --pwd ${NEW_GRAN_PATH} --env PYTHONPATH=${PYTHONPATH}:${NEW_GRAN_PATH} ${NEW_GRAN_PATH}/docker/image.sif python3 gran/rand/main.py
```
Run notebooks
```
# Start the notebook
apptainer exec --nv --bind ${NEW_GRAN_PATH}:${NEW_GRAN_PATH} --pwd ${NEW_GRAN_PATH} --env PYTHONPATH=${PYTHONPATH}:${NEW_GRAN_PATH} ${NEW_GRAN_PATH}/docker/image.sif jupyter notebook --allow-root --no-browser --port 1234
# Tunnel through SSH
ssh ginkgo -NL 1234:localhost:1234
```
