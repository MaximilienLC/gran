# Copyright 2023 The Gran Authors.
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

import glob
import os
import pickle
import random
import sys

from mpi4py import MPI
import numpy as np
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch


MAX_INT = 2**31 - 1


def evaluate():
    """
    Initialization of various MPI variables.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    config = OmegaConf.create(sys.argv[1])
    os.chdir(sys.argv[2])

    """
    Initialize environment
    """
    import gymnasium

    from gran.util.wrapper.gym_fb_control import (
        get_task_name,
        hide_score,
        run_env_step,
        reset_env_state,
    )

    env = gymnasium.make(get_task_name(config.task))

    """
    Distribute workload
    """
    folders = glob.glob("*")

    all_generations = [
        int(folder)
        for folder in folders
        if folder.isdigit() and os.path.isdir(folder)
    ]

    all_generations.sort()

    assigned_generations = [
        all_generations[i]
        for i in range(len(all_generations))
        if i % size == rank
    ]

    for gen in assigned_generations:
        print("Gen : " + str(gen))

        path = str(gen) + "/"

        # No state file
        if not os.path.isfile(path + "state.pkl"):
            print("No saved state found at " + os.getcwd() + path + ".")
            continue

        # Already evaluated
        if os.path.isfile(path + "evaluation.pkl"):
            continue

        with open(path + "state.pkl", "rb") as f:
            state = pickle.load(f)

        agents, fitnesses, total_num_env_steps = state

        fitnesses_sorting_indices = fitnesses.argsort(axis=0)
        fitnesses_rankings = fitnesses_sorting_indices.argsort(axis=0)
        selected = np.greater_equal(
            fitnesses_rankings, config.ecosystem.pop_size // 2
        )
        selected_indices = np.where(selected[:, 0] == True)[0]

        evaluation = [
            np.zeros((config.ecosystem.pop_size // 2, config.num_tests)),
            total_num_env_steps,
        ]

        assert not (config.seed == -1 and config.num_tests > 1)

        for i in range(config.ecosystem.pop_size // 2):
            agent = agents[selected_indices[i]][0]

            for j in range(config.num_tests):
                if config.seed == -1:
                    seed = MAX_INT - j
                else:
                    seed = config.seed

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                agent.reset()

                obs = reset_env_state(env, seed)

                for k in range(config.ecosystem.run_num_steps_per_test):
                    if "imitation" in config.env_path:
                        obs = hide_score(obs)

                    obs, rew, done = run_env_step(env, agent(obs))
                    evaluation[0][i][j] += rew

                    if done:
                        break

        with open(path + "evaluation.pkl", "wb") as f:
            pickle.dump(evaluation, f)
