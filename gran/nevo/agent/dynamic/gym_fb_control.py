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

from gran.nevo.agent.dynamic.base import BaseDynamicAgent
from gran.nevo.net.dynamic.recurrent import Net
from gran.util.math import RunningStandardization
from gran.util.wrapper.gym_fb_control import get_env_info
from gran.util.misc import config


class Agent(BaseDynamicAgent):
    """
    Gym Feature-Based Control Dynamic Agent.
    """

    def initialize__(self) -> None:

        if self.pop_idx == 0:
            self.type = "generator"
        else:  # self.pop_idx == 1:
            self.type = "discriminator"

        info = get_env_info(config.env)
        d_input, d_output, self.discrete_output, self.output_bound = info

        if config.agent.pop_merge:

            self.net = Net([d_input, d_output + 1])

        else:

            if self.type == "generator":
                self.net = Net([d_input, d_output])
            else:  # self.type == 'discriminator':
                self.net = Net([d_input, 1])

        self.feature_scaling = RunningStandardization()

    def __call__(self, x: np.ndarray) -> np.ndarray:

        x = self.env_to_net(x)
        x = self.net(x)
        x = self.net_to_env(x)

        return x

    def env_to_net(self, x: np.ndarray) -> np.ndarray:

        x = self.feature_scaling(x)

        return x

    def net_to_env(self, x: np.ndarray) -> np.ndarray:

        x = np.array(x).squeeze(axis=1)

        if config.agent.pop_merge:
            x = x[:-1] if self.type == "generator" else x[-1]  # discrim

        if self.type == "generator":

            if self.discrete_output:
                x = np.argmax(x)
            else:
                x = np.minimum(x, 1)
                x = x * (self.output_bound * 2) - self.output_bound

        else:  # self.type == 'discriminator':

            x = x.squeeze()
            x = np.minimum(x, 1)

        return x
