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

import numpy as np

from gran.rands.bots.cpu.base import CPUBotBase


class DynamicCPUBotBase(CPUBotBase):
    """
    Dynamic CPU Bot Base class.
    Concrete subclasses need to be named *Bot*.
    """

    def initialize_(self) -> None:
        """
        Initialize the bot's net's architecture.
        """
        self.initialize__()

        net.initialize_architecture()

    def initialize__(self) -> None:
        """
        Extra initialization step. Needs to be implemented.
        """
        assert hasattr(self, "net"), "Attribute 'net' required."

    def mutate_(self) -> None:
        """
        Mutation method for the dynamic bot. First initializes the nets'
        architectures. Then, every iteration, mutate network parameters and
        randomly select N mutations among all net mutations.
        """
        np.random.choice(self.nets_architectural_mutations)()
