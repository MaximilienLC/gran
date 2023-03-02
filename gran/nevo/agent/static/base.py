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

import torch

from gran.nevo.agent.base import BaseAgent
from gran.util.misc import config


class BaseStaticAgent(BaseAgent, metaclass=ABCMeta):
    """
    Base Static Agent class.
    Concrete subclasses need to be named *Agent*.
    """

    @final
    def initialize_(self) -> None:
        """
        Initialize the agent's mutation variance & nets.
        """
        self.initialize__()

        assert hasattr(self, "net")

        for param in self.net.parameters():
            param.requires_grad = False
            param.data = torch.zeros_like(param.data)

    @abstractmethod
    def initialize__(self) -> None:
        """
        Extra initialization step. Needs to be implemented.
        """
        pass

    @final
    def mutate_(self) -> None:
        """
        Mutation method for the static agent. Mutates the networks' parameters.
        """
        for param in self.net.parameters():
            param.data += config.ecosystem.sigma.value * torch.randn_like(
                param.data
            )
