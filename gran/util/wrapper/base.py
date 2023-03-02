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

import gymnasium

from gran.util.misc import config


class BaseWrapper(gymnasium.Wrapper):
    """
    .
    """

    def __init__(self, env):
        """
        .
        """
        super().__init__(env)

        if config.autoencoder.name != "none":
            self.autoencoder = instantiate(
                config.autoencoder
            ).load_from_checkpoint(config.autoencoder.checkpoint)

        if config.autoregressor.name != "none":
            self.autoregressor = instantiate(
                config.autoregressor
            ).load_from_checkpoint(config.autoregressor.checkpoint)

    def reset(self, seed):

        obs, info = self.env.reset(seed=seed)

        if config.autoencoder.name != "none":

            self.autoencoder.eval()

            with torch.no_grad():
                obs = self.autoencoder(obs)

        if config.autoregressor.name != "none":

            self.autoregressor.eval()
            self.autoregressor.reset()

            self.obs = obs

        return obs

    def step(self, action):
        """
        .
        """
        if config.autoregressor.name != "none":

            obs_action = torch.cat((self.obs, action)).view(1, 1, -1)

            with torch.no_grad():
                self.obs, rew, done = self.autoregressor(obs_action)

            self.obs = self.obs.cpu().squeeze().numpy()
            self.rew = self.rew.cpu().squeeze().numpy()
            self.done = bool(self.done.cpu().squeeze().numpy())

            return self.obs, rew, done, done, None

        else:

            obs, rew, term, trunc, info = self.env.step(action)

            if config.autoencoder.name != "none":

                with torch.no_grad():
                    obs = self.autoencoder(obs)

            return obs, rew, term or trunc

    def render(self):
        pass
