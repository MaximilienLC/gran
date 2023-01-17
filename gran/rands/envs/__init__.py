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

from abc import ABC, abstractmethod
from importlib import import_module

import numpy as np
from omegaconf import DictConfig


class BaseEnv(ABC):
    """
    Env Base class.
    Concrete subclasses need to be named *Env*.
    """

    def __init__(self, cfg: DictConfig, io_path: str, nb_pops: int) -> None:
        """
        Args:
            cfg - Experiment specific configuration.
            io_path - Path to the environment's IO class.
            nb_pops - Total number of populations.
        """
        self.cfg = cfg
        self.io_path = io_path
        self.nb_pops = nb_pops

        self.verify_task_and_transfer()

        self.initialize_io()
        self.initialize_bots()

        self.evaluates_on_gpu = False

    def verify_task_and_transfer(self) -> None:

        assert hasattr(self, "get_valid_tasks"), "Envs require the attribute "
        "'get_valid_tasks': a function that returns a list of all valid tasks."

        assert (
            self.cfg.env.task in self.get_valid_tasks()
        ), "'cfg.env.task' needs to be chosen from one of " + str(
            self.get_valid_tasks()
        )

        ge0_int = lambda x: isinstance(x, int) and x >= 0

        assert (
            ge0_int(self.cfg.env.seeding) or self.cfg.env.seeding == "reg"
        ), "'cfg.seeding' needs to be an int >= 0 or string 'reg'."

        assert ge0_int(self.cfg.env.steps), "'cfg.steps' needs to be an "
        "int >= 0."

        """
        transfer_options = ["none", "fit", "env+fit", "mem+env+fit"]

        assert self.cfg.transfer in transfer_options, ""
        "'cfg.transfer' needs be chosen from one of " + str(
            transfer_options
        )
        """

    def initialize_io(self) -> None:
        """
        Called upon object initialization.
        Initializes an IO object.
        IO objects handle file input/output.
        """
        self.io = getattr(import_module(self.io_path), "IO")(self.cfg)

    def initialize_bots(self) -> None:
        """
        Called upon object initialization.
        Initializes bots.
        Bots evolve and produce behaviour in the environment.
        """
        self.bots = []

        for pop_idx in range(self.nb_pops):
            self.bots.append(
                getattr(import_module(self.cfg.bots_path), "Bot")(
                    self.cfg, pop_idx, self.nb_pops
                )
            )

    def mutate_bots(
        self,
        pop_size: int,
        bots_batch_size: int,
        gen_nb: int,
        rank: int,
        bots_batch_index: int,
    ) -> None:
        """
        Mutate bots according to a seed created from .

        Args:
            pop_size - Population size.
            bots_batch_size - Batch size of bots maintained by MPI processes.
            gen_nb - Current generation number.
            rank - MPI process rank.
            bots_batch_index - Index of current bot in the batch of bots.
        """
        for pop_idx in range(self.nb_pops):
            self.bots[pop_idx].mutate(
                pop_size, bots_batch_size, gen_nb, rank, bots_batch_index
            )

    @abstractmethod
    def run_bots(self, gen_nb: int) -> np.ndarray:
        """
        Method called once per iteration in order to evaluate and attribute
        fitnesses to bots, all the while recording their number of steps.

        Args:
            gen_nb - Generation number.
        Returns:
            np.ndarray - Fitnesses and number of steps.
        """
        raise NotImplementedError
