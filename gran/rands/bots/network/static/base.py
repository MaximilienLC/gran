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

import torch

from gran.rands.bots.network.base import NetworkBotBase


class StaticNetworkBotBase(NetworkBotBase):
    """
    Static Network Bot Base class. These bot contains one static-sized
    PyTorch Neural Networks. Concrete subclasses need to be named *Bot*.
    """

    def initialize_bot(self) -> None:
        """
        Initialize the bot's mutation variance & nets.
        """
        self.sigma = 0.01

        for net in self.nets:

            for parameter in net.parameters():

                parameter.requires_grad = False
                parameter.data = torch.zeros_like(parameter.data)

    def _mutate(
        self,
        pop_size: int,
        bots_batch_size: int,
        gen_nb: int,
        rank: int,
        bots_batch_index: int,
    ) -> None:
        """
        Mutation method for the static bot.
        Mutates the networks' parameters.

        Args:
            pop_size - Population size.
            bots_batch_size - MPI process bots batch size.
            gen_nb - Generation number.
            rank - MPI rank.
            bots_batch_index - Bot batch index.
        """
        seed = gen_nb * pop_size + rank * bots_batch_size + bots_batch_index

        torch.manual_seed(seed)

        for net in self.nets:
            for parameter in net.parameters():
                parameter.data += self.sigma * torch.randn_like(parameter.data)
