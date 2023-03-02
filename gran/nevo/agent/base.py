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

from abc import ABCMeta, abstractmethod
from typing import final
import random
from typing import Any, final

import numpy as np
import torch

from gran.util.misc import config


class BaseAgent(metaclass=ABCMeta):
    """
    Base Agent class.
    Concrete subclasses need to be named *Agent*.
    """

    @final
    def __init__(self, pop_idx: int) -> None:
        """
        Args:
            pop_idx - Agent population idx.
        """
        self.pop_idx = pop_idx

    @final
    def initialize(self) -> None:
        """
        Initialize the agent before it starts getting built.
        """
        self.curr_run_score = 0
        self.curr_run_num_steps = 0

        if config.ecosystem.algorithm == "ga":

            if isinstance(config.ecosystem.gen_transfer, str):

                if "env" in config.ecosystem.gen_transfer:

                    self.saved_env_state = None
                    self.saved_env_obs = None
                    self.saved_env_seed = None

                    self.curr_episode_score = 0
                    self.curr_episode_num_steps = 0

                if "fit" in config.ecosystem.gen_transfer:

                    self.continual_fitness = 0

        self.initialize_()

    @abstractmethod
    def initialize_(self) -> None:
        """
        Extra initialization step. Needs to be implemented.
        """
        pass

    @final
    def reset(self) -> None:
        """
        Reset the agent and its net's inner state.
        """
        self.net.reset()

        self.reset_()

    def reset_(self) -> None:
        """
        Extra reset step. Can be implemented.
        """
        pass

    @final
    def mutate(self, seed: int, curr_gen: int) -> None:
        """
        Root mutate method. Calls a user-defined `mutate_` function after
        checking for potential initialization logic on generation #0.

        Args:
            seed - A random seed.
            curr_gen - Current generation number.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        if curr_gen == 0:
            self.initialize()

        self.mutate_()

    @abstractmethod
    def mutate_(self) -> None:
        """
        Method to mutate the agent. Needs to be implemented.
        """
        pass

    @abstractmethod
    def __call__(self, x) -> Any:
        """
        Run the agent for one timestep given input 'x'. Needs to be implemented.

        Args:
            x - Input
        Returns:
            Any - Output
        """
        pass
