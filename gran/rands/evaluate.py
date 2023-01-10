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

import argparse
import glob
import os
import pickle
import random
import sys

from mpi4py import MPI
import numpy as np
import torch

from gran.rands.utils.misc import mpi_fork


def main(args):

    """
    Initialization of various MPI-specific variables.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    MAX_INT = 2**31 - 1

    """
    Process arguments
    """
    if args.states_path[-1] == "/":
        args.states_path = args.states_path[:-1]

    split_path = args.states_path.split("/")

    env_path = split_path[-4]
    extra_arguments = split_path[-3]
    bots_path = split_path[-2]
    pop_size = int(split_path[-1])

    split_extra_arguments = extra_arguments.split("~")

    if "score" in env_path:

        seeding = split_extra_arguments[0].split(".")[1]
        steps = split_extra_arguments[1].split(".")[1]
        task = split_extra_arguments[2].split(".")[1]
        transfer = split_extra_arguments[3].split(".")[1]
        trials = split_extra_arguments[4].split(".")[1]

    elif "imitate.retro" in env_path:

        level = split_extra_arguments[0].split(".")[1]
        merge = split_extra_arguments[1].split(".")[1]
        steps = split_extra_arguments[2].split(".")[1]
        subject = split_extra_arguments[3].split(".")[1]
        task = split_extra_arguments[4].split(".")[1]
        transfer = split_extra_arguments[5].split(".")[1]

    else:  # 'imitate' in env_path:

        merge = split_extra_arguments[0].split(".")[1]
        steps = split_extra_arguments[1].split(".")[1]
        task = split_extra_arguments[2].split(".")[1]
        transfer = split_extra_arguments[3].split(".")[1]

    """
    Initialize environment
    """

    if "control" in env_path:

        import gym
        from gran.utils.control import get_control_task_name

        emulator = gym.make(get_control_task_name(task))

        hide_score = lambda x: x

    elif "atari" in env_path:

        import gym

        from gran.utils.atari import (
            get_atari_task_name,
            hide_atari_score as hide_score,
            wrap_atari,
        )

        emulator = wrap_atari(
            gym.make(
                get_atari_task_name(task),
                frameskip=1,
                repeat_action_probability=0,
            )
        )

    elif "retro" in env_path:

        import retro
        from utils.functions.retro import get_task_name, hide_score
        from utils.functions.retro import get_state_name

        emulator = retro.make(
            game=get_task_name(task), state=get_state_name(level)
        )

    else:  # 'gravity' in env_path:

        data = np.load("data/behaviour/gravity/11k.npy")[-1000:]

    """
    Import bots
    """

    if "control" in env_path:

        if "dynamic.rnn" in bots_path:
            from bots.netted.dynamic.rnn.control import Bot
        elif "static.rnn" in bots_path:
            from bots.netted.static.rnn.control import Bot
        else:  #'static.fc' in bots_path:
            from bots.netted.static.fc.control import Bot

    elif "atari" in env_path:

        if "dynamic" in bots_path:
            from bots.netted.dynamic.conv_rnn.atari import Bot
        else:  # 'static' in bots_path:
            from bots.netted.static.conv_rnn.atari import Bot

    elif "retro" in env_path:

        if "dynamic" in bots_path:
            from bots.netted.dynamic.conv_rnn.retro import Bot
        else:  # 'static' in bots_path:
            from bots.netted.static.conv_rnn.retro import Bot

    else:  # 'gravity' in env_path:

        if "dynamic.conv_rnn" in bots_path:
            from bots.netted.dynamic.conv_rnn.gravity import Bot
        else:  # 'static.conv_rnn' in bots_path:
            from bots.netted.static.conv_rnn.gravity import Bot

    """
    Distribute workload
    """

    files = [os.path.basename(x) for x in glob.glob(args.states_path + "/*")]

    gens = []

    for file in files:
        if file.isdigit() and os.path.isdir(args.states_path + "/" + file):
            gens.append(int(file))

    gens.sort()

    process_gens = []

    for i in range(len(gens)):
        if i % size == rank:
            process_gens.append(gens[i])

    for gen in process_gens:

        print("Gen : " + str(gen))

        path = args.states_path + "/" + str(gen) + "/"

        # Already evaluated
        if os.path.isfile(path + "scores.npy"):
            continue

        if not os.path.isfile(path + "state.pkl"):
            raise Exception("No saved state found at " + state_path + ".")

        with open(path + "state.pkl", "rb") as f:
            state = pickle.load(f)

        bots, fitnesses, total_nb_emulator_steps = state

        fitnesses_sorting_indices = fitnesses.argsort(axis=0)
        fitnesses_rankings = fitnesses_sorting_indices.argsort(axis=0)
        selected = np.greater_equal(fitnesses_rankings, pop_size // 2)
        selected_indices = np.where(selected[:, 0] == True)[0]

        if "gravity" in env_path:
            scores = np.zeros((pop_size // 2))
        else:
            scores = np.zeros((pop_size // 2, args.nb_tests))

        for i in range(pop_size // 2):

            bot = bots[selected_indices[i]][0]

            bot.setup_to_run()

            if seeding != "reg":
                args.nb_tests = 1

            for j in range(args.nb_tests):

                if seeding == "reg":
                    seed = MAX_INT - j
                else:
                    seed = int(seeding)

                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)
                bot.reset()

                if "gravity" not in env_path:

                    obs, _ = emulator.reset(seed=seed)

                else:  # 'gravity' in env_path:

                    data_point = data[j]
                    if task == "predict":
                        nb_obs_fed_to_generator = 3
                    else:
                        nb_obs_fed_to_generator = 1  # 'generate'

                for k in range(args.nb_obs_per_test):

                    if "gravity" not in env_path:

                        if "imitate" in env_path:
                            obs = hide_score(obs)

                        obs, rew, terminated, truncated, _ = emulator.step(
                            bot(obs)
                        )

                        scores[i][j] += rew

                        if terminated or truncated:
                            break

                    else:  # 'gravity' in env_path:

                        if k < nb_obs_fed_to_generator:
                            obs = data_point[k]

                        obs = bot(obs)

                        if torch.is_tensor(obs):
                            obs = obs.numpy()

                        scores[i] += np.sum((obs - data_point[k + 1]) ** 3)

                        if k == 2:
                            break

        np.save(path + "scores.npy", scores)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_mpi_processes",
        "-n",
        type=int,
        default=1,
        help="Number of MPI processes to fork and run simultaneously.",
    )

    parser.add_argument(
        "--states_path",
        "-s",
        type=str,
        required=True,
        help="Path to the saved states <=> "
        "data/states/<env_path>/<additional_arguments>/"
        "<bots_path>/<population_size>/",
    )

    parser.add_argument(
        "--nb_tests",
        "-t",
        type=int,
        default=10,
        help="Number of tests to evaluate the agents on.",
    )

    parser.add_argument(
        "--nb_obs_per_test",
        "-o",
        type=int,
        default=2**31 - 1,
        help="Number of observations per test.",
    )

    args = parser.parse_args()

    is_forking_process = mpi_fork(args.nb_mpi_processes)

    if is_forking_process:
        sys.exit()

    main(args)
