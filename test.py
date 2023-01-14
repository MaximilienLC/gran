import os
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path=".", config_name="config")
def fork(cfg: DictConfig):

    import subprocess

    env = os.environ.copy()
    env.update(MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", INSIDE_MPI_FORK="1")

    # --allow-run-as-root needed to run inside Docker
    subprocess.check_call(
        [
            "mpiexec",
            "--allow-run-as-root",
            "-n",
            str(cfg.rands.nb_mpi_processes),
            sys.executable,
            sys.argv[0],
            str(cfg),
            str(HydraConfig.get().runtime.output_dir),
        ],
        env=env,
    )


def main():

    from mpi4py import MPI

    cfg = OmegaConf.create(sys.argv[1])
    os.chdir(sys.argv[2])

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(rank, size, cfg)


if __name__ == "__main__":

    if os.getenv("INSIDE_MPI_FORK"):
        main()
    else:
        fork()
