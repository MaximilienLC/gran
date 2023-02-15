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
import torch.nn as nn
import torch.nn.functional as F

from gran.grad.model.base import BaseModel


class BaseAEModel(BaseModel):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        """
        Args:
            encoder: Encoder model
            decoder: Decoder model
        """
        super().__init__()

        self.encoder, self.decoder = encoder, decoder

        assert hasattr(self, "latent_shape"), "AE Models require attribute "
        "'latent_shape': the shape of the latent code."

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x         batch_size x (x_shape)
        Returns:
            recon_x   batch_size x (x_shape)
        """
        latent = self.encoder(x)

        recon_x = self.decoder(latent)

        return recon_x

    def step(self, batch):
        """
        Args:
            batch     batch_size x (x_shape)

        Returns:
            loss      1
        """
        x = batch
        recon_x = self(x)
        loss = self.compute_loss(x, recon_x)

        return loss

    def compute_loss(self, x, recon_x):
        """
        Args:
            x         batch_size x (x_shape)
            recon_x   batch_size x (x_shape)

        Returns:
            loss      1
        """
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")

        return recon_loss
