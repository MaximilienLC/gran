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
import subprocess
import sys

import cv2
import numpy as np


def grayscale_rescale_divide(x: np.ndarray, d_in: int) -> np.ndarray:

    x = x.squeeze()

    if x.ndim == 3:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)

    x = cv2.resize(x, (d_in, d_in), interpolation=cv2.INTER_AREA)

    x = x.astype(np.float32)

    x = x / 255

    return x


def mpi_fork(n):
    """
    Re-launches the current script through mpiexec with N processes.
    The forking process returns True, others return False.
    """
    if not os.getenv("INSIDE_MPI_FORK"):

        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", INSIDE_MPI_FORK="1"
        )

        # --allow-run-as-root needed to run inside Docker
        subprocess.check_call(
            ["mpiexec", "--allow-run-as-root", "-n", str(n), sys.executable]
            + ["-u"]
            + sys.argv,
            env=env,
        )

        return True

    else:

        return False


def update_running_mean_std(x, old_mean, old_var, n):

    new_mean = old_mean + (x - old_mean) / n
    new_var = old_var + (x - old_mean) * (x - new_mean)
    new_std = np.sqrt(new_var / n)

    return new_mean, new_var, new_std
