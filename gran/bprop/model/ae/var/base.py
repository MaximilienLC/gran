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

from typing import Tuple

import torch
import torch.nn.functional as F

from gran.bprop.model.ae.base import BaseAEModel
from gran.util.misc import cfg


class BaseVAEModel(BaseAEModel):
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x         batch_size x (x_shape)
        Config:
            beta:     beta parameter for KL loss
        Returns:
            recon_x   batch_size x (x_shape)
            mu        batch_size x (latent_shape)
            log_sigma batch_size x (latent_shape)
        """
        mu, log_sigma = self.encoder(x).split(self.latent_shape[0], dim=1)
        sigma = torch.exp(log_sigma)

        epsilon = torch.randn(self.latent_shape, device=self.device)

        z = mu + sigma * epsilon

        recon_x = self.decoder(z)

        return recon_x, mu, log_sigma

    def step(self, batch):
        x = batch
        recon_x, mu, log_sigma = self(x)
        loss = self.compute_loss(x, recon_x, mu, log_sigma)

        return loss

    def compute_loss(self, x, recon_x, mu, log_sigma):
        """
        Args:
            x         batch_size x (x_shape)
            recon_x   batch_size x (x_shape)
            mu        batch_size x (latent_shape)
            log_sigma batch_size x (latent_shape)
        """
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")

        kl_loss = -0.5 * torch.sum(
            1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp()
        )

        self.log("train/recon_loss", recon_loss)
        self.log("train/kl_loss", kl_loss)

        return recon_loss + cfg.beta * kl_loss
