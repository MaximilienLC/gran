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
from typing import Any

import numpy as np
from omegaconf import DictConfig
import random
import torch


class BaseBot(ABC):
    """
    Base Bot class.
    Concrete subclasses need to be named *Bot*.
    """

    def __init__(self, cfg: DictConfig, pop_idx: int, nb_pops: int) -> None:
        """
        Args:
            cfg - Experiment specific configuration.
            pop_idx - Population index.
            nb_pops - Total number of populations.
        """
        self.cfg = cfg
        self.pop_idx = pop_idx
        self.nb_pops = nb_pops

    def initialize(self) -> None:
        """
        Initialize the bot before it starts getting built.
        """
        self.current_run_score = 0
        self.current_run_nb_steps = 0

        if self.cfg.alg == "ga":

            if "env" in self.cfg.env.transfer:

                self.saved_emulator_state = None
                self.saved_emulator_obs = None
                self.saved_emulator_seed = None

                self.current_episode_score = 0
                self.current_episode_nb_steps = 0

            if "fit" in self.cfg.env.transfer:

                self.continual_fitness = 0

        self.initialize_()

    def initialize_(self) -> None:
        """
        Extra initialization step. Needs to be implemented.
        """
        assert hasattr(self, "net"), "Attribute 'net' required."

    def reset(self) -> None:
        """
        Reset the bot and its nets' inner states.
        """
        self.net.reset()

        self.reset_()

    def reset_(self) -> None:
        """
        Extra reset step. Can be implemented or left blank if this
        functionality is not required.
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
        Root mutate method. Calls a user-defined `mutate_` function after
        checking for potential initialization logic on generation #0.

        Args:
            pop_size - Population size.
            bots_batch_size - Batch size of bots maintained by MPI processes.
            gen_nb - Current generation number.
            rank - MPI process rank.
            bots_batch_index - Index of current bot in the batch.
        """
        seed = gen_nb * pop_size + rank * bots_batch_size + bots_batch_index

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if gen_nb == 0:
            self.initialize()

        self.mutate_()

    @abstractmethod
    def mutate_(self) -> None:
        """
        Method to mutate the bot. Needs to be implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, x) -> Any:
        """
        Run the bot for one timestep given input 'x'. Needs to be implemented.

        Args:
            x - Input
        Returns:
            Any - Output
        """
        raise NotImplementedError
