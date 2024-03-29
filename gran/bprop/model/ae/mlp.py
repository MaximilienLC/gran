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

import torch.nn as nn

from gran.bprop.model.ae.base import BaseAEModel


class MLPAEModel(BaseAEModel):
    """
    Multilayer perceptron Autoencoder Model
    """

    def __init__(self, x_size: int) -> None:
        """
        Args:
            config: Config.
            x_size: Number of input features
        Config:
            hidden_size: Number of hidden neurons
            latent_size: Size of latent vector
        """
        encoder = nn.Sequential(
            nn.Linear(x_size, config.encoder.hidden_size),
            nn.ReLU(),
            nn.Linear(config.encoder.hidden_size, config.encoder.latent_size),
        )

        decoder = nn.Sequential(
            nn.Linear(config.encoder.latent_size, config.encoder.hidden_size),
            nn.ReLU(),
            nn.Linear(config.encoder.hidden_size, x_size),
        )

        self.latent_shape = (config.encoder.latent_size,)

        super().__init__(encoder, decoder)
