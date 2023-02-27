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

from gran.ne.agent.dynamic.base import BaseDynamicBot
from gran.ne.net.dynamic.original import Net
from gran.util.wrapper.gym_fb_control import get_task_info
from gran.util.misc import config, RunningStandardization


class Agent(BaseDynamicBot):
    """
    Dynamic Agent class.
    """

    def initialize__(self) -> None:
        """
        Initialize the agent's network and various variables.
        """
        if self.pop_idx == 0:
            self.type = "generator"
        else:  # self.pop_idx == 1:
            self.type = "discriminator"

        info = get_task_info(config.task)
        d_input, d_output, self.discrete_output, self.output_bound = info

        if config.ecosystem.merge:
            self.net = Net([d_input, d_output + 1])

        else:
            if self.type == "generator":
                self.net = Net([d_input, d_output])
            else:  # self.type == 'discriminator':
                self.net = Net([d_input, 1])

        # Setup variables for a running standardization of inputs
        self.feature_scaling = RunningStandardization()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        """
        x = self.env_to_net(x)
        x = self.net(x)
        x = self.net_to_env(x)

        return x

    def env_to_net(self, x: np.ndarray) -> np.ndarray:
        """
        Update the inputs received from the environment.
        """
        x = self.feature_scaling(x)

        return x

    def net_to_env(self, x: np.ndarray) -> np.ndarray:
        """
        Update the outputs to send back to the environment.
        """
        x = np.array(x).squeeze(axis=1)

        if config.ecosystem.merge:
            x = x[:-1] if self.type == "generator" else x[-1]  # discrim

        if self.type == "generator":
            if self.discrete_output:
                x = np.argmax(x)
            else:
                x *= self.output_bound

        else:  # self.type == 'discriminator':
            x = x.squeeze()
            x = (x + 1) / 2

        return x
