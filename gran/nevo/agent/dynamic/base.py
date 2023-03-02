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

from abc import ABCMeta, abstractmethod
from typing import final

import numpy as np

from gran.nevo.agent.base import BaseAgent
from gran.util.misc import config


class BaseDynamicAgent(BaseAgent, metaclass=ABCMeta):
    """
    Base Dynamic Agent class.
    Concrete subclasses need to be named *Bot*.
    """

    @final
    def initialize_(self) -> None:
        """
        Initialize the agent's net's architecture.
        """
        self.initialize__()

        assert hasattr(self, "net")

        self.net.initialize_architecture()

        self.architectural_mutation_probability_logits = np.ones(
            len(self.net.architectural_operations)
        )

        self.num_architectural_mutations = 1

    @abstractmethod
    def initialize__(self) -> None:
        """
        Extra initialization step. Needs to be implemented.
        """
        pass

    @final
    def mutate_(self) -> None:
        """
        Mutation method for the dynamic agent. Randomly select an architectural
        mutation.
        """
        if config.ecosystem.sigma.is_adaptive:

            # 1. Mutate probabilities
            idx = np.random.randint(len(self.net.architectural_operations))
            op = np.random.choice([np.multiply, np.divide])

            self.architectural_mutation_probability_logits[idx] = op(
                self.architectural_mutation_probability_logits[idx], 2
            )

            # 2. Mutate number of sampling steps
            minus_1_or_plus_1 = np.random.choice([-1, 1])

            self.num_architectural_mutations += int(minus_1_or_plus_1)

            if self.num_architectural_mutations < 1:
                self.num_architectural_mutations = 1

        probs = self.architectural_mutation_probability_logits / np.sum(
            self.architectural_mutation_probability_logits
        )

        for _ in range(self.num_architectural_mutations):
            np.random.choice(self.net.architectural_operations, p=probs)()
