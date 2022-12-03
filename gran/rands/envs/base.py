# Copyright 2022 Maximilien Le Clei.
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
import argparse
from importlib import import_module

import numpy as np


class EnvBase(ABC):
    """
    Env Base class. Environments are virtual playgrounds for bots to evolve
    and produce behaviour. One environment makes interact, at a given time, as
    many bots as there are populations.
    Concrete subclasses need to be named *Env*.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        io_path: str,
        nb_pops: int,
    ):
        """
        Constructor.

        Args:
            args - Experiment specific arguments (obtained through argparse).
            io_path - Path to the environment's IO class.
            nb_pops - Total number of populations.
        """
        self.args = args
        self.io_path = io_path
        self.nb_pops = nb_pops

        self.initialize_io()
        self.initialize_bots()

        self.evaluates_on_gpu = False

    def initialize_io(self) -> None:
        """
        Called upon object initialization.
        Initializes an IO object.
        IO objects handle file input/output.
        """
        self.io = getattr(import_module(self.io_path), "IO")(self.args)

    def initialize_bots(self) -> None:
        """
        Called upon object initialization.
        Initializes bots.
        Bots evolve and produce behaviour in the environment.
        """
        bot_import_path = self.args.bots_path.replace("/", ".").replace(
            ".py", ""
        )

        self.bots = []

        for pop_nb in range(self.nb_pops):
            self.bots.append(
                getattr(import_module(bot_import_path), "Bot")(
                    self.args, pop_nb, self.nb_pops
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
        for pop_nb in range(self.nb_pops):
            self.bots[pop_nb].mutate(
                pop_size, bots_batch_size, gen_nb, rank, bots_batch_index
            )

    def setup_to_run(self) -> None:
        """
        Setup bots to later run them within the environment.
        """
        for bot in self.bots:
            bot.setup_to_run()

    def setup_to_save(self) -> None:
        """
        Setup bots to then later save them (pickling to a file or pickling to
        be sent to another process).
        """
        for bot in self.bots:
            bot.setup_to_save()

    def evaluate_bots(self, gen_nb: int) -> np.ndarray:
        """
        Method called once per iteration in order to evaluate, record number
        of steps and attribute fitnesses to bots.

        Args:
            gen_nb - Generation number.
        Returns:
            np.ndarray - Numpy array of fitnesses and number of steps.
        """
        self.setup_to_run()
        fitnesses_and_nb_steps = self.run(gen_nb)
        self.setup_to_save()

        return fitnesses_and_nb_steps

    @abstractmethod
    def run(self, gen_nb: int) -> np.ndarray:
        """
        Inner method of *evaluate_bots*.

        Args:
            gen_nb - Generation number.
        Returns:
            np.ndarray - Numpy array of fitnesses.
        """
        raise NotImplementedError()
