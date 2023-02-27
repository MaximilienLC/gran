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

from gran.ne.agent.base import BaseBot


class BaseDynamicBot(BaseBot):
    """
    Base Dynamic Agent class.
    Concrete subclasses need to be named *Bot*.
    """

    def initialize_(self) -> None:
        """
        Initialize the agent's net's architecture.
        """
        self.initialize__()

        self.net.initialize_architecture()

    def initialize__(self) -> None:
        """
        Extra initialization step. Needs to be implemented.
        """
        assert hasattr(self, "net"), "Attribute 'net' required."

    def mutate_(self) -> None:
        """
        Mutation method for the dynamic agent. Randomly select an architectural
        mutation.
        """
        np.random.choice(self.net.architectural_mutations)()
