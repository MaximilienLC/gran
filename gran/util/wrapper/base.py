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

import glob

import gymnasium
from hydra.utils import instantiate
import torch

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

        self.autoencoding = (
            config.autoencoder.name != "none" and config.stage != "collect"
        )

        self.autoregressing = (
            config.autoregressor.name != "none" and config.stage != "collect"
        )

        if self.autoencoding:

            checkpoint = glob.glob("ae/*.ckpt")

            self.autoencoder = instantiate(
                config.autoencoder
            ).load_from_checkpoint(checkpoint[0])

        if self.autoregressing:

            checkpoint = glob.glob(
                "ar/lightning_logs/version_0/checkpoints/*.ckpt"
            )

            self.autoregressor = instantiate(
                config.autoregressor
            ).load_from_checkpoint(checkpoint[0])

    def reset(self, seed):

        obs, _ = self.env.reset(seed=seed)

        if self.autoencoding:

            self.autoencoder.eval()

            with torch.no_grad():
                obs = self.autoencoder(obs)

        if self.autoregressing:

            self.autoregressor.eval()
            self.autoregressor.reset()

            self.obs = obs

        return obs

    def step(self, action):
        """
        .
        """
        if self.autoregressing:

            self.obs = torch.tensor(self.obs).to(self.autoregressor.device)
            action = torch.tensor([1, 0] if action == 0 else [0, 1]).to(
                self.autoregressor.device
            )

            obs_action = torch.cat((self.obs, action)).view(1, 1, -1)

            with torch.no_grad():
                self.obs, rew, done = self.autoregressor(obs_action)

            self.obs = self.obs.cpu().squeeze().numpy()
            rew = rew.cpu().squeeze().numpy()
            done = bool(done.cpu().squeeze().numpy())

            print(self.obs)
            return self.obs, rew, done

        else:

            obs, rew, term, trunc, _ = self.env.step(action)

            if self.autoencoding:

                with torch.no_grad():
                    obs = self.autoencoder(obs)

            return obs, rew, term or trunc

    def render(self):
        pass
