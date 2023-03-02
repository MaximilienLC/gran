"""
Training agents in environments.
"""

import os
import subprocess
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="agent")
def main(config: DictConfig):

    for curr_iter in range(config.num_iters):

        if config.stage == "train":

            if config.autoencoder.name != "none":

                autoencoder = instantiate(config.autoencoder)
                autoencoder.train()

            if config.autoregressor.name != "none":

                autoregressor = instantiate(config.autoregressor)
                autoregressor.train()

        fork_neuroevolution(curr_iter, config.num_cpus)


def fork_neuroevolution(curr_iter: int, num_cpus: int):

    from gran.util.misc import update_config_for_nevo

    update_config_for_nevo(curr_iter)

    env = os.environ.copy()

    env.update(MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", INSIDE_MPI_FORK="1")

    subprocess.check_call(
        [
            "mpiexec",
            "--allow-run-as-root",  # Docker
            "--oversubscribe",  # Submitit
            "-n",
            str(num_cpus),
            sys.executable,
            "-m",
            "gran",
        ],
        env=env,
    )


if __name__ == "__main__":

    if not os.getenv("INSIDE_MPI_FORK"):

        main()

    else:

        from gran.util.misc import config

        if config.stage == "train":

            from gran.nevo.train import train

            train()

        else:  # config.stage == "test":

            from gran.nevo.test import test

            test()
