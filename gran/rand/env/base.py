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

from gran.util.misc import cfg


class BaseEnv(ABC):
    """
    Env Base class.
    Concrete subclasses need to be named *Env*.
    """

    def __init__(self, io_path: str, num_pops: int) -> None:
        """
        Args:
            io_path - Path to the environment's IO class.
            num_pops - Total number of populations.
        """
        self.io_path = io_path
        self.num_pops = num_pops

        self.verify_task_and_transfer()

        self.initialize_io()
        self.initialize_bots()

        self.evaluates_on_gpu = False

    def verify_task_and_transfer(self) -> None:

        assert hasattr(self, "valid_tasks"), "Envs require the attribute "
        "'valid_tasks': a list of all valid tasks."

        assert cfg.task in self.valid_tasks

        assert isinstance(cfg.seed, int) and cfg.seed >= -1
        assert isinstance(cfg.num_steps, int) and cfg.seed >= -1

        # transfer_options = ["none", "fit", "env+fit", "mem+env+fit"]

        # assert cfg.transfer in transfer_options

    def initialize_io(self) -> None:
        """
        Called upon object initialization.
        Initializes an IO object.
        IO objects handle file input/output.
        """
        self.io = getattr(import_module(self.io_path), "IO")()

    def initialize_bots(self) -> None:
        """
        Called upon object initialization.
        Initializes bots.
        Bots evolve and produce behaviour in the environment.
        """
        self.bots = [None] * self.num_pops

        for pop_idx in range(self.num_pops):
            self.bots[pop_idx] = getattr(import_module(cfg.bot_path), "Bot")(
                pop_idx
            )

    def mutate_bots(self, seeds: int, curr_gen: int) -> None:
        """
        Mutate bots according to a seed created from .

        Args:
            seeds - List of random seeds.
        """
        for pop_idx in range(self.num_pops):
            self.bots[pop_idx].mutate(seeds[pop_idx], curr_gen)

    @abstractmethod
    def run_bots(self, curr_gen: int) -> np.ndarray:
        """
        Method called once per iteration in order to evaluate and attribute
        fitnesses to bots, all the while recording their number of steps.

        Args:
            curr_gen - Current generation.
        Returns:
            np.ndarray - Fitnesses and number of steps.
        """
        raise NotImplementedError
