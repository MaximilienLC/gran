# Gran: Grads, Rands & Neural Nets

## Setting up the repository

### Step 1. Building the Docker / Apptainer images locally

(If you are in the `rrg-pbellec` Alliance Canada group you can skip Step 1)  
  
The following instructions define the full dependency installation steps on **Ubuntu** to ensure reproducibility across platforms.  
We only outline the full-guide with container technologies Docker & Apptainer (Singularity). 
However you can probably install this repository in fewer steps if softwares like CUDA/cuDNN and Python packages are already installed on your system.  
For the sake of simplicity & reproducibility, we however recommend you follow this guide.

#### i. Install and setup Docker (skip if already setup)
Install the deb Docker package (can also install from the official website: [[link](https://docs.docker.com/engine/install/)])
```
sudo apt-get install -y docker.io
```
Give yourself docker permissions to not require sudo privileges
```
sudo groupadd docker
sudo usermod -aG docker ${USER}
```
Finally, log out and log back to apply changes

#### ii. Install the NVIDIA Driver (skip if already installed)
1) "Show Applications"  
2) "Software & Updates"  
3) "Additional Drivers"  
4) Select: "Using NVIDIA driver metapackage from nvidia-driver-XXX (proprietery, tested)"  
  
Installation for other mediums: [[link](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)]

#### iii. Install the NVIDIA Container Toolkit (skip if already installed)

Setup the package repository and the GPG key
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
Install the package and restart the docker engine
```
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
  
Installation for other mediums: [[link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)]

#### iv. Clone the repository and build the Docker image
```
git clone git@github.com:MaximilienLC/gran.git
cd gran/
docker build -f docker/Dockerfile -t gran:latest .
```

#### v. Apptainer (skip if you have sudo privileges on your cloud cluster)

##### Install
```
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt-get update
sudo apt-get install -y apptainer
```

##### Convert the Docker image to a Singularity/Apptainer file
```
docker save gran:latest -o image.tar
apptainer build image.sif docker-archive://image.tar
```

### Step 2. Setting up the SLURM-based cluster (skip if not applicable)

#### i. Clone the repository and set-up the virtualenv to run Submitit jobs
Clone & install Python packages
```
cd ${SCRATCH}
git clone git@github.com:MaximilienLC/gran.git ${SCRATCH}/gran/
module load python/3.10
python3 -m venv ${SCRATCH}/gran/venv
. ${SCRATCH}/gran/venv/bin/activate
pip install -r ${SCRATCH}/gran/reqs/hydra_reqs.txt
```
Hard-coding Submitit fix to run with Apptainer
```
sed -i "s|shlex.quote(sys.executable)|\"apptainer exec --nv --bind ${SCRATCH}\/gran:${SCRATCH}\/gran --pwd ${SCRATCH}\/gran ${SCRATCH}\/gran\/image.sif --env PYTHONPATH=\${PYTHONPATH}:${SCRATCH}\/gran python3\"|" ${SCRATCH}/gran/venv/lib/python3.10/site-packages/submitit/slurm/slurm.py
```

#### ii. Copy over the Apptainer image
```
LOCAL_USER=max
AC_USER=mleclei
AC_CLUSTER=beluga.computecanada.ca
scp ${AC_USER}@${AC_CLUSTER}:/scratch/image.tar .

cp /scratch/mleclei/gran/image.tar /scratch/${AC_USER}/gran/image.tar
cp /scratch/mleclei/gran/image.sif /scratch/${AC_USER}/gran/image.sif
```

### Step 3. Execute the sample code
```
tmux
cd ${SCRATCH}/gran
. venv/bin/activate
pip install -e .
python3 gran/rand/main.py
```


### Optional step 1. Build a container from the local Docker image
#### Copy back the 


```
#                       MPI in Docker   JNotebook                                                              RAM access
docker run --name granc --privileged -p 8888:8888 --gpus=all -it -v /home/max/Dropbox/gran:/gran -w /gran -v /dev/shm/:/dev/shm/  gran:latest
```

### Optional step 2. Build a container from the local Docker image
```
jupyter notebook --allow-root --ip 0.0.0.0
```