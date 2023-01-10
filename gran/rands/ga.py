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
Genetic algorithm of the Grads, Rands & Nets library (a.k.a. Gran).
Forked and executed by as many CPU processes as are provided through
`cfg.rands.nb_mpi_processes`.

CPU processes communicate at regular intervals through `mpi4py` send, gather &
scatter operations

At its core, this genetic algorithm loops over three stages: mutation,
evaluation & selection. The computational workload of the mutation stage is
spread across all CPU processes. The work required for the evaluation stage is
then either scattered across all CPU processes, or instead sent to specific
CPU processes for 1-to-1 CPU-to-GPU communication (in the case that GPU devices
are provided and compatible with the given task). Finally, the selection stage
is performed by the main process.
"""
import copy
from importlib import import_module
import os
import pickle
import subprocess
import sys
import time

import hydra
from hydra.core.hydra_config import HydraConfig
from mpi4py import MPI
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb


@hydra.main(
    version_base=None, config_path="../../config", config_name="config"
)
def fork(cfg: DictConfig):

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

    cfg = OmegaConf.create(sys.argv[1])
    os.chdir(sys.argv[2])

    """
    Initialization of various MPI & GA-specific variables.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    nb_gpus = torch.cuda.device_count()

    assert cfg.rands.population_size % size == 0, ""
    "'cfg.rands.population_size' must be a multiple of "
    "'cfg.rands.nb_mpi_processes'."

    if cfg.rands.wandb_logging:

        os.environ["WANDB_SILENT"] = "true"

        with open("wandb_key.txt", "r") as f:
            key = f.read()

        wandb.login(key=key)

        wandb_group_id = wandb.util.generate_id() if rank == 0 else None
        wandb_group_id = comm.bcast(wandb_group_id, root=0)
        wandb.init(group=wandb_group_id)

    env = getattr(import_module(cfg.rands.env_path), "Env")(cfg)

    old_nb_gen = cfg.rands.nb_elapsed_generations
    new_nb_gen = cfg.rands.nb_generations
    pop_size = cfg.rands.population_size
    bots_batch_size = pop_size // size
    nb_pops = env.nb_pops

    bots = None
    bots_batch = []

    # [MPI buffer size, pair position, sending]
    pairings = None
    pairings_batch = np.empty((bots_batch_size, nb_pops, 3), dtype=np.uint32)

    # [fitnesses, # emulator steps]
    fitnesses_and_nb_emulator_steps_batch = np.empty(
        (bots_batch_size, nb_pops, 2), dtype=np.float32
    )

    # [fitnesses, # emulator steps, pickled bot sizes]
    generation_results = None
    generation_results_batch = np.empty(
        (bots_batch_size, nb_pops, 3), dtype=np.float32
    )

    if rank == 0:

        fitnesses = np.empty((pop_size, nb_pops), dtype=np.float32)

        pairings = np.empty((pop_size, nb_pops, 3), dtype=np.uint32)

        generation_results = np.empty((pop_size, nb_pops, 3), dtype=np.float32)

        total_nb_emulator_steps = 0

    if old_nb_gen > 0:

        """
        Primary process loads previous experiment state.
        """
        if rank == 0:

            state = env.io.load_state()

            fitnesses, total_nb_emulator_steps, bots = state

            fitnesses_sorting_indices = fitnesses.argsort(axis=0)
            fitnesses_rankings = fitnesses_sorting_indices.argsort(axis=0)

            bots = [
                bots[i * bots_batch_size : (i + 1) * bots_batch_size]
                for i in range(size)
            ]

        """
        Primary process scatters bots from the previous experiment
        to secondary processes.
        """
        bots_batch = comm.scatter(bots, root=0)

    """
    Setting up of GPU-evaluation specific MPI communication.
    """
    if env.evaluates_on_gpu:

        gpu_queueing_group_bots_batch = []

        fitnesses_and_nb_emulator_steps_gpu_queuer_batch = np.empty(
            (pop_size // nb_gpus, nb_pops, 2), dtype=np.float32
        )

        nb_cpu_procs_per_gpu = size // nb_gpus
        is_gpu_queuing_cpu_proc = rank % nb_cpu_procs_per_gpu == 0
        gpu_queueing_group_id = rank // nb_cpu_procs_per_gpu

        gpu_queueing_group_proc_list = np.arange(
            gpu_queueing_group_id * nb_cpu_procs_per_gpu,
            (gpu_queueing_group_id + 1) * nb_cpu_procs_per_gpu,
        ).tolist()

        gpu_queueing_group_comm = comm.Create_group(
            comm.group.Incl(gpu_queueing_group_proc_list)
        )

    """
    Start of the genetic algorithm.
    """
    for gen_nb in range(old_nb_gen, old_nb_gen + new_nb_gen):

        if rank == 0:
            start = time.time()

        """
        Primary process generates peer-to-peer pairing information
        and scatters to secondary processes.
        """
        if gen_nb > 0:

            if rank == 0:

                # MPI buffer size
                pairings[:, :, 0] = np.max(generation_results[:, :, 2])

                for j in range(nb_pops):

                    pair_ranking = (
                        fitnesses_rankings[:, j] + pop_size // 2
                    ) % pop_size

                    # pair position
                    pairings[:, j, 1] = fitnesses_sorting_indices[:, j][
                        pair_ranking
                    ]

                # sending
                pairings[:, :, 2] = np.greater_equal(
                    fitnesses_rankings, pop_size // 2
                )

            comm.Scatter(pairings, pairings_batch, root=0)

            """
            Processes exchange bots 
            """
            req = []

            for i in range(bots_batch_size):

                for j in range(nb_pops):

                    pair = int(pairings_batch[i, j, 1] // bots_batch_size)

                    if pairings_batch[i, j, 2] == 1:  # sending

                        tag = int(pop_size * j + bots_batch_size * rank + i)

                        req.append(
                            comm.isend(bots_batch[i][j], dest=pair, tag=tag)
                        )

                    else:  # pairing_and_seeds_batch[i, j, 2] == 0: # receiving

                        tag = int(pop_size * j + pairings_batch[i, j, 1])

                        req.append(
                            comm.irecv(
                                pairings_batch[i, j, 0],
                                source=pair,
                                tag=tag,
                            )
                        )

            received_bots = MPI.Request.waitall(req)

            for i, bot in enumerate(received_bots):
                if bot is not None:
                    bots_batch[i // nb_pops][i % nb_pops] = bot

        """
        Variation.
        """
        for i in range(bots_batch_size):

            if gen_nb > 0:
                env.bots = bots_batch[i]

            env.mutate_bots(pop_size, bots_batch_size, gen_nb, rank, i)

            """
            CPU Evaluation.
            """
            if not env.evaluates_on_gpu:

                fitnesses_and_nb_emulator_steps_batch[i] = env.run_bots(gen_nb)

                if gen_nb == 0:
                    bots_batch.append(copy.deepcopy(env.bots))

        """
        GPU Evaluation.
        """
        if env.evaluates_on_gpu:

            gpu_queueing_group_bots_batch = gpu_queueing_group_comm.gather(
                bots_batch,
                root=gpu_queueing_group_id * nb_cpu_procs_per_gpu,
            )

            if is_gpu_queuing_cpu_proc:

                fitnesses_and_nb_emulator_steps_gpu_queuer_batch = (
                    env.run_bots(gpu_queueing_group_bots_batch, gen_nb)
                )

            gpu_queueing_group_comm.Scatter(
                fitnesses_and_nb_emulator_steps_gpu_queuer_batch,
                fitnesses_and_nb_emulator_steps_batch,
                root=gpu_queueing_group_id * nb_cpu_procs_per_gpu,
            )

            bots_batch = gpu_queueing_group_comm.scatter(
                gpu_group_bots_batch,
                root=gpu_group_id * nb_cpu_procs_per_gpu,
            )

        generation_results_batch[
            :, :, 0:2
        ] = fitnesses_and_nb_emulator_steps_batch

        for i in range(bots_batch_size):
            for j in range(nb_pops):
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

            if cfg.rands.merge == "yes":
                fitnesses[:, 0] += fitnesses[:, 1][::-1]
                fitnesses[:, 1] = fitnesses[:, 0][::-1]

            fitnesses_sorting_indices = fitnesses.argsort(axis=0)
            fitnesses_rankings = fitnesses_sorting_indices.argsort(axis=0)

            print(
                gen_nb + 1,
                ":",
                int(time.time() - start),
                "\n",
                np.mean(fitnesses, 0),
                "\n",
                np.max(fitnesses, 0),
            )

            nb_emulator_steps = generation_results[:, :, 1]
            total_nb_emulator_steps += nb_emulator_steps.sum()

        """
        State saving
        """
        if gen_nb + 1 in env.io.save_points or gen_nb == 0:

            batched_bots = comm.gather(bots_batch, root=0)

            if rank == 0:

                bots = []

                for bot_batch in batched_bots:
                    bots = bots + bot_batch

                env.io.save_state(
                    [bots, fitnesses, total_nb_emulator_steps], gen_nb + 1
                )

    wandb.finish()


if __name__ == "__main__":

    if os.getenv("INSIDE_MPI_FORK"):
        main()
    else:
        fork()
