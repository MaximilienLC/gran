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

import copy
from importlib import import_module
import os
import pickle
import sys
import time
import warnings

from mpi4py import MPI
import numpy as np
from omegaconf import OmegaConf
import torch
import wandb


def evolve():
    """
    Initialization of various MPI & EA-specific variables.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_gpus = torch.cuda.device_count()

    cfg = OmegaConf.create(sys.argv[1])

    assert cfg.pop_size % cfg.num_mpi_procs == 0

    assert not (cfg.alg == "es" and cfg.transfer != "none")
    assert not (cfg.alg == "es" and cfg.merge == "yes")

    np.set_printoptions(suppress=True)
    warnings.filterwarnings("ignore")
    sys.setrecursionlimit(2**31 - 1)

    if cfg.wandb_logging:

        os.environ["WANDB_SILENT"] = "true"

        with open("wandb_key.txt", "r") as f:
            key = f.read()

        wandb.login(key=key)

        wandb_group_id = wandb.util.generate_id() if rank == 0 else None
        wandb_group_id = comm.bcast(wandb_group_id, root=0)
        wandb.init(group=wandb_group_id)

    os.chdir(sys.argv[2])  # Change working directory to hydra working dir

    from gran.rand.util.misc import generate_new_seeds

    env = getattr(import_module(cfg.env_path), "Env")()

    bots = None
    bots_batch = []
    bots_batch_size = cfg.pop_size // size

    if cfg.alg == "ga":
        # [MPI buffer size, pair position, sending, seed]
        exchange_and_mutate_info = None
        exchange_and_mutate_info_batch = np.empty(
            (bots_batch_size, env.num_pops, 4), dtype=np.uint32
        )

    # [fitnesses, # emulator steps]
    fitnesses_and_num_emulator_steps_batch = np.empty(
        (bots_batch_size, env.num_pops, 2), dtype=np.float32
    )

    # [fitnesses, # emulator steps, pickled bot sizes]
    generation_results = None
    generation_results_batch = np.empty(
        (bots_batch_size, env.num_pops, 3), dtype=np.float32
    )

    if rank == 0:

        if cfg.alg == "ga":
            exchange_and_mutate_info = np.empty(
                (cfg.pop_size, env.num_pops, 4), dtype=np.uint32
            )

        fitnesses = np.empty((cfg.pop_size, env.num_pops), dtype=np.float32)

        generation_results = np.empty(
            (cfg.pop_size, env.num_pops, 3), dtype=np.float32
        )

        total_num_emulator_steps = 0

    if cfg.prev_num_gens > 0:

        """
        Primary process loads previous experiment state.
        """
        if rank == 0:

            state = env.io.load_state()

            bots, fitnesses, total_num_emulator_steps = state

            fitnesses_sorting_indices = fitnesses.argsort(axis=0)
            fitnesses_rankings = fitnesses_sorting_indices.argsort(axis=0)

            bots = [
                bots[i * bots_batch_size : (i + 1) * bots_batch_size]
                for i in range(size)
            ]

        """
        Primary process scatters bots from the previous experiment
        to secondary procs.
        """
        bots_batch = comm.scatter(bots, root=0)

    """
    Setting up of GPU-evaluation specific MPI communication.
    """
    if env.evaluates_on_gpu:

        num_cpu_procs_per_gpu = size // num_gpus
        gpu_idx = rank // num_cpu_procs_per_gpu
        is_gpu_queuing_cpu_proc = rank % num_cpu_procs_per_gpu == 0

        ith_gpu_bots_batch = []
        ith_gpu_bots_batch_size = cfg.pop_size // num_gpus

        ith_gpu_cpu_proc_list = np.arange(
            gpu_idx * num_cpu_procs_per_gpu,
            (gpu_idx + 1) * num_cpu_procs_per_gpu,
        ).tolist()

        ith_gpu_fitnesses_and_num_emulator_steps_batch = np.empty(
            (ith_gpu_bots_batch_size, env.num_pops, 2), dtype=np.float32
        )

        ith_gpu_comm = comm.Create_group(
            comm.group.Incl(ith_gpu_cpu_proc_list)
        )

    """
    Start of the genetic algorithm.
    """
    for curr_gen in range(cfg.prev_num_gens, cfg.prev_num_gens + cfg.num_gens):

        np.random.seed(curr_gen)

        if rank == 0:

            start = time.time()

            if cfg.alg == "ga":

                new_seeds = generate_new_seeds(curr_gen, env.num_pops)

                if curr_gen != 0:
                    for j in range(env.num_pops):
                        new_seeds[:, j] = new_seeds[:, j][
                            fitnesses_rankings[:, j]
                        ]

        """
        ES: Primary process broadcasts new weights to all secondary processes.
        GA: Primary process generates peer-to-peer exchange information
        and scatters to secondary processes.
        """
        if curr_gen > 0:

            if cfg.alg == "es":

                comm.Bcast(new_weights, root=0)

            else:  # if cfg.alg == "ga":

                if rank == 0:

                    # MPI buffer size
                    exchange_and_mutate_info[:, :, 0] = np.max(
                        generation_results[:, :, 2]
                    )

                    for j in range(env.num_pops):

                        pair_ranking = (
                            fitnesses_rankings[:, j] + cfg.pop_size // 2
                        ) % cfg.pop_size

                        # pair position
                        exchange_and_mutate_info[
                            :, j, 1
                        ] = fitnesses_sorting_indices[:, j][pair_ranking]

                    # sending
                    exchange_and_mutate_info[:, :, 2] = np.greater_equal(
                        fitnesses_rankings, cfg.pop_size // 2
                    )

                    # seeds
                    exchange_and_mutate_info[:, :, 3] = new_seeds

                comm.Scatter(
                    exchange_and_mutate_info,
                    exchange_and_mutate_info_batch,
                    root=0,
                )

                """
                Processes exchange bots 
                """
                req = []

                for i in range(bots_batch_size):

                    for j in range(env.num_pops):

                        pair = int(
                            exchange_and_mutate_info_batch[i, j, 1]
                            // bots_batch_size
                        )

                        # sending
                        if exchange_and_mutate_info_batch[i, j, 2] == 1:

                            tag = int(
                                cfg.pop_size * j + bots_batch_size * rank + i
                            )

                            req.append(
                                comm.isend(
                                    bots_batch[i][j], dest=pair, tag=tag
                                )
                            )

                        # receiving
                        else:

                            tag = int(
                                cfg.pop_size * j
                                + exchange_and_mutate_info_batch[i, j, 1]
                            )

                            req.append(
                                comm.irecv(
                                    exchange_and_mutate_info_batch[i, j, 0],
                                    source=pair,
                                    tag=tag,
                                )
                            )

                received_bots = MPI.Request.waitall(req)

                for i, bot in enumerate(received_bots):
                    if bot is not None:
                        bots_batch[i // env.num_pops][i % env.num_pops] = bot

        """
        Variation.
        """
        for i in range(bots_batch_size):

            if curr_gen > 0:
                env.bots = bots_batch[i]

            if cfg.alg == "ga":
                seeds = exchange_and_mutate_info_batch[i, :, 3]
            else:  # cfg.alg == "es":
                seeds = np.random.randint(2**32, size=2, dtype=np.uint32)

            env.mutate_bots(seeds, curr_gen)

            """
            CPU Evaluation.
            """
            if env.evaluates_on_gpu == False:

                fitnesses_and_num_emulator_steps_batch[i] = env.run_bots(
                    curr_gen
                )

                if curr_gen == 0:
                    bots_batch.append(copy.deepcopy(env.bots))

        """
        GPU Evaluation.
        """
        if env.evaluates_on_gpu:

            ith_gpu_bots_batch = ith_gpu_comm.gather(
                bots_batch,
                root=gpu_idx * num_cpu_procs_per_gpu,
            )

            if is_gpu_queuing_cpu_proc:

                ith_gpu_fitnesses_and_num_emulator_steps_batch = env.run_bots(
                    ith_gpu_bots_batch, curr_gen
                )

            ith_gpu_comm.Scatter(
                ith_gpu_fitnesses_and_num_emulator_steps_batch,
                fitnesses_and_num_emulator_steps_batch,
                root=gpu_idx * num_cpu_procs_per_gpu,
            )

            if cfg.transfer != "none":
                bots_batch = ith_gpu_comm.scatter(
                    ith_gpu_bots_batch,
                    root=gpu_idx * num_cpu_procs_per_gpu,
                )

        generation_results_batch[
            :, :, 0:2
        ] = fitnesses_and_num_emulator_steps_batch

        for i in range(bots_batch_size):
            for j in range(env.num_pops):
                generation_results_batch[i, j, 2] = len(
                    pickle.dumps(bots_batch[i][j])
                )

        """
        Primary process gathers fitnesses, emulator steps and pickled bot sizes
        """
        comm.Gather(generation_results_batch, generation_results, root=0)

        """
        Fitness processing.
        """
        if rank == 0:

            fitnesses = generation_results[:, :, 0]

            if cfg.merge == "yes":
                fitnesses[:, 0] += fitnesses[:, 1][::-1]
                fitnesses[:, 1] = fitnesses[:, 0][::-1]

            fitnesses_sorting_indices = fitnesses.argsort(axis=0)
            fitnesses_rankings = fitnesses_sorting_indices.argsort(axis=0)

            num_emulator_steps = generation_results[:, :, 1]
            total_num_emulator_steps += num_emulator_steps.sum()

            print(curr_gen + 1, ":", int(time.time() - start))
            print(np.mean(fitnesses, 0), "\n", np.max(fitnesses, 0))

        """
        State saving
        """
        if curr_gen + 1 in env.io.save_points or curr_gen == 0:

            batched_bots = comm.gather(bots_batch, root=0)

            if rank == 0:

                bots = []

                for bot_batch in batched_bots:
                    bots = bots + bot_batch

                env.io.save_state(
                    [bots, fitnesses, total_num_emulator_steps], curr_gen + 1
                )

    wandb.finish()
