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

import torch

from gran.rand.bot.base import BaseBot
from gran.util.misc import cfg


class BaseStaticBot(BaseBot):
    """
    Base Static Bot class.
    Concrete subclasses need to be named *Bot*.
    """

    def initialize_(self) -> None:
        """
        Initialize the bot's mutation variance & nets.
        """
        self.initialize__()

        for param in self.net.parameters():
            param.requires_grad = False
            param.data = torch.zeros_like(param.data)

    def initialize__(self) -> None:
        """
        Extra initialization step. Needs to be implemented.
        """
        assert hasattr(self, "net"), "Attribute 'net' required."

    def mutate_(self) -> None:
        """
        Mutation method for the static bot.
        Mutates the networks' parameters.
        """
        for param in self.net.parameters():
            param.data += cfg.sigma * torch.randn_like(param.data)
