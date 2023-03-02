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

import copy
from importlib import import_module
import os
import pickle
import sys
import time
import warnings

from mpi4py import MPI
import numpy as np
import torch
import wandb

from gran.nevo.util.misc import generate_new_seeds
from gran.util.misc import config, get_task_domain

np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")
sys.setrecursionlimit(2**31 - 1)


def train():

    if config.wandb.mode != "disabled":

        os.environ["WANDB_SILENT"] = "true"

        with open("../../../../../../../../wandb_key.txt", "r") as f:
            key = f.read()

        wandb.login(key=key)

        wandb_group_id = wandb.util.generate_id() if rank == 0 else None
        wandb_group_id = comm.bcast(wandb_group_id, root=0)
        wandb.init(group=wandb_group_id)

    """
    Initialization of various MPI & EA-specific variables.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_gpus = torch.cuda.device_count()

    assert config.ecosystem.pop_size % config.num_cpus == 0

    if config.ecosystem.algorithm == "es":
        assert not config.ecosystem.gen_transfer
        assert not config.ecosystem.pop_merge

    space_path = (
        f"gran.nevo.space.{config.mode}.{get_task_domain(config.task)}"
    )

    space = getattr(import_module(space_path), "Space")()

    agents = None
    agents_batch = []
    agents_batch_size = config.ecosystem.pop_size // size

    if config.ecosystem.algorithm == "ga":
        # [MPI buffer size, pair position, sending, seed]
        exchange_and_mutate_info = None
        exchange_and_mutate_info_batch = np.empty(
            (agents_batch_size, space.num_pops, 4), dtype=np.uint32
        )

    # [fitnesses, # env steps]
    fitnesses_and_num_env_steps_batch = np.empty(
        (agents_batch_size, space.num_pops, 2), dtype=np.float32
    )

    # [fitnesses, # env steps, pickled agent sizes]
    generation_results = None
    generation_results_batch = np.empty(
        (agents_batch_size, space.num_pops, 3), dtype=np.float32
    )

    if rank == 0:
        if config.ecosystem.algorithm == "ga":
            exchange_and_mutate_info = np.empty(
                (config.ecosystem.pop_size, space.num_pops, 4), dtype=np.uint32
            )

        fitnesses = np.empty(
            (config.ecosystem.pop_size, space.num_pops), dtype=np.float32
        )

        generation_results = np.empty(
            (config.ecosystem.pop_size, space.num_pops, 3), dtype=np.float32
        )

        total_num_env_steps = 0

    if space.io.prev_num_gens > 0:
        """
        Primary process loads previous experiment state.
        """
        if rank == 0:

            state = space.io.load_state()

            agents, fitnesses, total_num_env_steps = state

            fitnesses_sorting_indices = fitnesses.argsort(axis=0)
            fitnesses_rankings = fitnesses_sorting_indices.argsort(axis=0)

            agents = [
                agents[i * agents_batch_size : (i + 1) * agents_batch_size]
                for i in range(size)
            ]

        """
        Primary process scatters agents from the previous experiment
        to secondary procs.
        """
        agents_batch = comm.scatter(agents, root=0)

    """
    Setting up of GPU-evaluation specific MPI communication.
    """
    if space.evaluates_on_gpu:

        num_cpu_procs_per_gpu = size // num_gpus
        gpu_idx = rank // num_cpu_procs_per_gpu
        is_gpu_queuing_cpu_proc = rank % num_cpu_procs_per_gpu == 0

        ith_gpu_agents_batch = []
        ith_gpu_agents_batch_size = config.ecosystem.pop_size // num_gpus

        ith_gpu_cpu_proc_list = np.arange(
            gpu_idx * num_cpu_procs_per_gpu,
            (gpu_idx + 1) * num_cpu_procs_per_gpu,
        ).tolist()

        ith_gpu_fitnesses_and_num_env_steps_batch = np.empty(
            (ith_gpu_agents_batch_size, space.num_pops, 2), dtype=np.float32
        )

        ith_gpu_comm = comm.Create_group(
            comm.group.Incl(ith_gpu_cpu_proc_list)
        )

    """
    Start of the genetic algorithm.
    """
    for curr_gen in range(
        space.io.prev_num_gens, space.io.prev_num_gens + space.io.new_num_gens
    ):

        np.random.seed(curr_gen)

        if rank == 0:

            start = time.time()

            if config.ecosystem.algorithm == "ga":
                new_seeds = generate_new_seeds(curr_gen, space.num_pops)

                if curr_gen != 0:
                    for j in range(space.num_pops):
                        new_seeds[:, j] = new_seeds[:, j][
                            fitnesses_rankings[:, j]
                        ]

        """
        ES: Primary process broadcasts new weights to all secondary processes.
        GA: Primary process generates peer-to-peer exchange information
        and scatters to secondary processes.
        """
        if curr_gen > 0:

            if config.ecosystem.algorithm == "es":
                comm.Bcast(new_weights, root=0)

            else:  # if config.ecosystem.algorithm == "ga":

                if rank == 0:

                    # MPI buffer size
                    exchange_and_mutate_info[:, :, 0] = np.max(
                        generation_results[:, :, 2]
                    )

                    for j in range(space.num_pops):

                        pair_ranking = (
                            fitnesses_rankings[:, j]
                            + config.ecosystem.pop_size // 2
                        ) % config.ecosystem.pop_size

                        # pair position
                        exchange_and_mutate_info[
                            :, j, 1
                        ] = fitnesses_sorting_indices[:, j][pair_ranking]

                    # sending
                    exchange_and_mutate_info[:, :, 2] = np.greater_equal(
                        fitnesses_rankings, config.ecosystem.pop_size // 2
                    )

                    # seeds
                    exchange_and_mutate_info[:, :, 3] = new_seeds

                comm.Scatter(
                    exchange_and_mutate_info,
                    exchange_and_mutate_info_batch,
                    root=0,
                )

                """
                Processes exchange agents 
                """
                req = []

                for i in range(agents_batch_size):
                    for j in range(space.num_pops):

                        pair = int(
                            exchange_and_mutate_info_batch[i, j, 1]
                            // agents_batch_size
                        )

                        # sending
                        if exchange_and_mutate_info_batch[i, j, 2] == 1:

                            tag = int(
                                config.ecosystem.pop_size * j
                                + agents_batch_size * rank
                                + i
                            )

                            req.append(
                                comm.isend(
                                    agents_batch[i][j], dest=pair, tag=tag
                                )
                            )

                        # receiving
                        else:

                            tag = int(
                                config.ecosystem.pop_size * j
                                + exchange_and_mutate_info_batch[i, j, 1]
                            )

                            req.append(
                                comm.irecv(
                                    exchange_and_mutate_info_batch[i, j, 0],
                                    source=pair,
                                    tag=tag,
                                )
                            )

                received_agents = MPI.Request.waitall(req)

                for i, agent in enumerate(received_agents):

                    if agent is not None:

                        agents_batch[i // space.num_pops][
                            i % space.num_pops
                        ] = agent

        """
        Variation.
        """
        for i in range(agents_batch_size):

            if curr_gen > 0:
                space.agents = agents_batch[i]

            if config.ecosystem.algorithm == "ga":
                seeds = exchange_and_mutate_info_batch[i, :, 3]
            else:  # config.ecosystem.algorithm == "es":
                seeds = np.random.randint(2**32, size=2, dtype=np.uint32)

            space.mutate_agents(seeds, curr_gen)

            """
            CPU Evaluation.
            """
            if space.evaluates_on_gpu == False:

                fitnesses_and_num_env_steps_batch[i] = space.run_agents(
                    curr_gen
                )

                if curr_gen == 0:
                    agents_batch.append(copy.deepcopy(space.agents))

        """
        GPU Evaluation.
        """
        if space.evaluates_on_gpu:

            ith_gpu_agents_batch = ith_gpu_comm.gather(
                agents_batch,
                root=gpu_idx * num_cpu_procs_per_gpu,
            )

            if is_gpu_queuing_cpu_proc:
                ith_gpu_fitnesses_and_num_env_steps_batch = space.run_agents(
                    ith_gpu_agents_batch, curr_gen
                )

            ith_gpu_comm.Scatter(
                ith_gpu_fitnesses_and_num_env_steps_batch,
                fitnesses_and_num_env_steps_batch,
                root=gpu_idx * num_cpu_procs_per_gpu,
            )

            if config.ecosystem.gen_transfer != "none":
                agents_batch = ith_gpu_comm.scatter(
                    ith_gpu_agents_batch,
                    root=gpu_idx * num_cpu_procs_per_gpu,
                )

        generation_results_batch[:, :, 0:2] = fitnesses_and_num_env_steps_batch

        for i in range(agents_batch_size):
            for j in range(space.num_pops):
                generation_results_batch[i, j, 2] = len(
                    pickle.dumps(agents_batch[i][j])
                )

        """
        Primary process gathers fitnesses, env steps and pickled agent sizes
        """
        comm.Gather(generation_results_batch, generation_results, root=0)

        """
        Fitness processing.
        """
        if rank == 0:

            fitnesses = generation_results[:, :, 0]

            if config.ecosystem.pop_merge == "yes":
                fitnesses[:, 0] += fitnesses[:, 1][::-1]
                fitnesses[:, 1] = fitnesses[:, 0][::-1]

            fitnesses_sorting_indices = fitnesses.argsort(axis=0)
            fitnesses_rankings = fitnesses_sorting_indices.argsort(axis=0)

            num_env_steps = generation_results[:, :, 1]
            total_num_env_steps += num_env_steps.sum()

            print(f"{curr_gen+1}: {int(time.time() - start)}")
            print(f"{np.mean(fitnesses, 0)}\n{np.max(fitnesses, 0)}")

        """
        State saving
        """
        if curr_gen + 1 in space.io.save_points or curr_gen == 0:

            batched_agents = comm.gather(agents_batch, root=0)

            if rank == 0:
                agents = []

                for agent_batch in batched_agents:
                    agents = agents + agent_batch

                space.io.save_state(
                    [agents, fitnesses, total_num_env_steps], curr_gen + 1
                )

    wandb.finish()
