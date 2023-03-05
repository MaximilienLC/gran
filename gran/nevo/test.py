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
import sys
import warnings

from mpi4py import MPI
import numpy as np

from gran.util.misc import config
from gran.util.wrapper.gym_fb_control import GymFeatureBasedControlWrapper

np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")
sys.setrecursionlimit(2**31 - 1)


def test():

    if config.wandb != "disabled":

        os.environ["WANDB_SILENT"] = "true"

        with open("../../../../../../../../wandb_key.txt", "r") as f:
            key = f.read()

        wandb.login(key=key)

        wandb_group_id = wandb.util.generate_id() if rank == 0 else None
        wandb_group_id = comm.bcast(wandb_group_id, root=0)
        wandb.init(group=wandb_group_id)

    MAX_INT = 2**31 - 1

    if config.agent.num_steps_per_test == "infinite":
        config.agent.num_steps_per_test = MAX_INT

    """
    Initialization of various MPI & EA-specific variables.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    """
    Initialize environment
    """
    env = GymFeatureBasedControlWrapper()

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
            fitnesses_rankings, config.agent.pop_size // 2
        )
        selected_indices = np.where(selected[:, 0] == True)[0]

        evaluation = [
            np.zeros(
                (config.agent.pop_size // 2, config.agent.num_tests)
            ),
            total_num_env_steps,
        ]

        for i in range(config.agent.pop_size // 2):

            agent = agents[selected_indices[i]][0]

            for j in range(config.agent.num_tests):

                seed_everything(MAX_INT - j)

                agent.reset()

                obs = env.reset(seed)

                num_steps = 

                for k in range(config.agent.num_steps_per_test):

                    obs, rew, done = env.step(agent(obs))
                    evaluation[0][i][j] += rew

                    if done:
                        break

        with open(path + "evaluation.pkl", "wb") as f:
            pickle.dump(evaluation, f)
