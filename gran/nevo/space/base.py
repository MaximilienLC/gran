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

from abc import ABC, abstractmethod
from importlib import import_module

import numpy as np

from gran.util.misc import cfg


class BaseSpace(ABC):
    """
    Space Base class. Spaces are virtual playgrounds for bots to evolve
    and produce behaviour. A space makes interact, at a given time,
    as many bots as there are populations.
    Concrete subclasses need to be named *Space*.
    """

    def __init__(self, io_path: str, num_pops: int) -> None:
        """
        Args:
            io_path - Path to the space's IO class.
            num_pops - Total number of populations.
        """
        self.io_path = io_path
        self.num_pops = num_pops

        self.verify_assertions()

        self.initialize_io()
        self.initialize_agents()

        self.evaluates_on_gpu = False

    def verify_assertions(self) -> None:
        """
        Verify assertions.
        """
        assert hasattr(self, "valid_tasks"), "Attribute 'valid_tasks' "
        "required: a list of all valid tasks."

        assert cfg.ecosystem.task in self.valid_tasks

        assert (
            isinstance(cfg.seed, int) and cfg.seed >= 0
        ) or cfg.seed == "scheduled"

        assert (
            isinstance(cfg.ecosystem.run_num_steps, int)
            and cfg.ecosystem.run_num_steps >= 0
        )

        assert cfg.ecosystem.gen_transfer in [
            "none",
            "fit",
            "env+fit",
            "mem+env+fit",
        ]

    def initialize_io(self) -> None:
        """
        Initializes an IO object. IO objects handle file input/output.
        """
        self.io = getattr(import_module(self.io_path), "IO")()

    def initialize_agents(self) -> None:
        """
        Initializes agents. Agents evolve and produce behaviour in the space.
        """
        self.agents = []

        for pop_idx in range(self.num_pops):
            self.agents.append(
                getattr(import_module(cfg.ecosystem.agent_path), "Agent")(
                    pop_idx
                )
            )

    def mutate_agents(self, seeds: int, curr_gen: int) -> None:
        """
        Mutate agents according to a seed created from .

        Args:
            seeds - List of random seeds.
            curr_gen - Current generation.
        """
        for pop_idx in range(self.num_pops):
            self.agents[pop_idx].mutate(seeds[pop_idx], curr_gen)

    @abstractmethod
    def run_agents(self, curr_gen: int) -> np.ndarray:
        """
        Method called once per iteration in order to evaluate and attribute
        fitnesses to agents, all the while recording their number of steps.

        Args:
            curr_gen - Current generation.
        Returns:
            np.ndarray - Fitnesses and number of steps.
        """
        raise NotImplementedError
