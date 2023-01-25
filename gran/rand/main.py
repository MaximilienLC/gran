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

import os
import sys
import subprocess

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from gran.rand.evolve import evolve
from gran.rand.evaluate import evaluate


@hydra.main(version_base=None, config_path="../../config", config_name="rand")
def fork(cfg: DictConfig):

    env = os.environ.copy()
    env.update(MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", INSIDE_MPI_FORK="1")

    subprocess.check_call(
        [
            "mpiexec",
            "--allow-run-as-root",  # Docker
            "--oversubscribe",  # Submitit
            "-n",
            str(cfg.num_mpi_procs),
            sys.executable,
            script,
            str(cfg),
            str(HydraConfig.get().runtime.output_dir),
        ],
        env=env,
    )


if __name__ == "__main__":

    if os.getenv("INSIDE_MPI_FORK"):

        cfg = OmegaConf.create(sys.argv[1])

        assert cfg.stage in ["train", "test"]

        if cfg.stage == "train":
            evolve()
        else:  # cfg.stage == "test":
            evaluate()

    else:

        global script
        script = sys.argv[0]

        fork()
