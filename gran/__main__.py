# Copyright 2022 The Gran Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Training agents in environments.
"""

import os
import subprocess
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="agent")
def main(config: DictConfig):

    autoenconding = config.autoencoder.name != "none"
    autoregressing = config.autoregressor.name != "none"

    if config.stage == "train":

        for curr_iter in range(config.num_iters):

            # if autoenconding or autoregressing:
            #     fork_mpi_processes("collect", curr_iter, config.num_cpus)

            # if autoenconding:

            #     autoencoder = instantiate(config.autoencoder)
            #     autoencoder.train()

            # if autoregressing:

            #     from gran.bprop.train import train

            #     train()

            if config.agent.mode == "neuroevolution":
                fork_mpi_processes("train", curr_iter, config.num_cpus)
            else:
                agent = instantiate(config.agent)
                agent.train()

    else:  # config.stage == "test":

        if config.agent.mode == "neuroevolution":
            fork_mpi_processes("test", curr_iter, config.num_cpus)
        else:
            agent = instantiate(config.agent)
            agent.test()


def fork_mpi_processes(stage: str, curr_iter: int, num_cpus: int):

    from gran.util.misc import update_config_for_fork

    update_config_for_fork("pre-fork", stage, curr_iter)

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

    update_config_for_fork("post-fork", stage, curr_iter)


if __name__ == "__main__":

    if not os.getenv("INSIDE_MPI_FORK"):

        main()

    else:

        from gran.util.misc import config

        if config.stage == "collect":

            from gran.nevo.collect import collect

            collect()

        elif config.stage == "train":

            from gran.nevo.train import train

            train()

        else:  # config.stage == "test":

            from gran.nevo.test import test

            test()
