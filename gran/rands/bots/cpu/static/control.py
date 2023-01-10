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

import numpy as np
import torch

from gran.rands.bots.cpu.static.base import StaticCPUBotBase
from gran.rands.nets.static.cpu import Net
from gran.utils.control import get_control_task_info
from gran.rands.utils.misc import update_running_mean_std


class Bot(StaticCPUBotBase):
    def initialize__(self):

        if self.pop_idx == 0:
            self.function = "generator"
        else:  # self.pop_idx == 1:
            self.function = "discriminator"

        info = get_control_task_info(self.cfg.rands.env.task)
        d_input, d_output, self.discrete_output, self.output_bound = info
        recurrent = self.cfg.rands.bots.static.recurrent

        if self.cfg.rands.merge:

            self.net = Net(
                [d_input, 50, 50, d_output + 1], recurrent=recurrent
            )

        else:

            if self.function == "generator":
                self.net = Net(
                    [d_input, 50, 50, d_output], recurrent=recurrent
                )
            else:  # self.function == 'discriminator':
                self.net = Net([d_input, 50, 50, 1], recurrent=recurrent)

        # Setup variables for a running standardization of inputs
        self.mean = self.var = self.std = self.n = 0

    def __call__(self, x):

        x = self.env_to_net(x)
        x = self.net(x)
        x = self.net_to_env(x)

        return x

    def env_to_net(self, x):

        self.n += 1
        self.mean, self.var, self.std = update_running_mean_std(
            x, self.mean, self.var, self.n
        )

        x = (x - self.mean) / (self.std + (self.std == 0))

        x = x[None, :]
        x = torch.Tensor(x)

        return x

    def net_to_env(self, x):

        x = x.numpy().squeeze(axis=0)

        if self.cfg.rands.merge:
            x = x[:-1] if self.function == "generator" else x[-1]  # discrim

        if self.function == "generator":

            if self.discrete_output:
                x = np.argmax(x)
            else:
                x *= self.output_bound

        else:  # self.function == 'discriminator':

            x = x.squeeze()
            x = (x + 1) / 2

        return x
