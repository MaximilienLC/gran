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

from abc import abstractmethod
import argparse
from typing import Any


class BotBase:
    """
    Bot Base class. Bots are artificial agents that can both be randomly
    mutated and generate behaviour/predictions in environments.
    Concrete subclasses need to be named *Bot*.
    """

    def __init__(self, args: argparse.Namespace, pop_nb: int, nb_pops: int):
        """
        Constructor.

        Args:
            args - Experiment specific arguments (obtained through argparse).
            pop_nb - Population number.
            nb_pops - Total number of populations.
        """
        self.args = args
        self.pop_nb = pop_nb
        self.nb_pops = nb_pops

    def initialize(self) -> None:
        """
        Initialize the bot before it starts getting built. Can be implemented
        or left blank if this function is not necessary.
        """
        pass

    def mutate(
        self,
        pop_size: int,
        bots_batch_size: int,
        gen_nb: int,
        rank: int,
        bots_batch_index: int,
    ) -> None:
        """
        Root mutate method. Calls a user-defined _mutate function after
        checking for potential initialization on generation 0.

        Args:
            pop_size - Population size.
            bots_batch_size - Batch size of bots maintained by MPI processes.
            gen_nb - Current generation number.
            rank - MPI process rank.
            bot_batch_index - Index of current bot in the batch of bots.
        """
        if gen_nb == 0:
            self.initialize()

        self._mutate(pop_size, bots_batch_size, gen_nb, rank, bots_batch_index)

    @abstractmethod
    def _mutate(
        self,
        pop_size: int,
        bots_batch_size: int,
        gen_nb: int,
        rank: int,
        bots_batch_index: int,
    ) -> None:
        """
        Method to mutate the bot. Needs to be implemented.

        Args:
            pop_size - Population size.
            bots_batch_size - Batch size of bots maintained by MPI processes.
            gen_nb - Current generation number.
            rank - MPI process rank.
            bots_batch_index - Index of current bot in the batch of bots.
        """
        raise NotImplementedError()

    def setup_to_run(self) -> None:
        """
        Setup the bot to run in an environment. Can be implemented or left
        blank if this function is not necessary.
        """
        pass

    def setup_to_save(self) -> None:
        """
        Setup the bot to be saved (pickled to a file or to be sent to
        another process). Can be implemented or left blank if this function
        is not necessary.
        """
        pass

    def reset(self) -> None:
        """
        Reset the bot. Can be implemented or left blank if this function
        is not necessary.
        """
        pass

    @abstractmethod
    def __call__(self, x) -> Any:
        """
        Run the bot for one timestep given input 'x'. Needs to be implemented.

        Args:
            x - Input
        Returns:
            Any - Output
        """
        raise NotImplementedError()
