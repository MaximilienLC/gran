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

from gran.grad.model.ae.var.base import BaseVAEModel


class MLPVAE(BaseVAEModel):
    def __init__(
        self, x_size: int, hidden_size: int, latent_size: int
    ) -> None:
        """
        Args:
            x_size: Number of input features
            hidden_size: Number of hidden neurons
            latent_size: Size of latent vector
        """
        encoder = nn.Sequential(
            nn.Linear(x_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size * 2),
        )

        decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, x_size),
        )

        self.latent_shape = (latent_size,)

        super().__init__(encoder, decoder)
