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

import pytorch_lightning as pl
import torch
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F

from gran.grads.models.ar import BaseARModel


class MDNRNN(pl.LightningModule):
    def __init__(
        self,
        latent_size: int,
        num_actions: int,
        hidden_size: int,
        num_gaussians: int,
        predicting_reward: bool = True,
        predicting_termination: bool = True,
    ) -> None:

        super().__init__()

        self.latent_size = latent_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians

        self.predicting_reward = predicting_reward
        self.predicting_termination = predicting_termination

        self.lstm = nn.LSTM(latent_size + num_actions, hidden_size)
        self.fc = nn.Linear(
            hidden_size, (1 + 2 * latent_size) * num_gaussians + 2
        )

    def forward(self, latent, action):
        """
        Args:
            latent    seq_len x batch_size x latent_size
            action    seq_len x batch_size x num_actions
        Returns:
            log_pi    seq_len x batch_size x num_gaussians
            mu        seq_len x batch_size x num_gaussians x latent_size
            sigma     seq_len x batch_size x num_gaussians x latent_size
            rew_hat   seq_len x batch_size
            done_hat  seq_len x batch_size
        """
        seq_len, batch_size = latent.size(0), latent.size(1)

        latent_and_action = torch.cat([latent, action], dim=-1)

        lstm_out, _ = self.lstm(latent_and_action)
        fc_out = self.linear(lstm_out)

        raw_pi, mu, log_sigma, rew_hat, done_hat = fc_out.split(
            [
                self.num_gaussians,
                self.num_gaussians * self.latent_size,
                self.num_gaussians * self.latent_size,
                1,
                1,
            ],
            dim=-1,
        )

        log_pi = F.log_softmax(
            raw_pi.view(seq_len, batch_size, self.num_gaussians), dim=-1
        )

        mu = mu.view(seq_len, batch_size, self.num_gaussians, self.latent_size)

        sigma = torch.exp(
            log_sigma.view(
                seq_len, batch_size, self.num_gaussians, self.latent_size
            )
        )

        return log_pi, mu, sigma, rew_hat, done_hat

    def step(self, batch):

        latent, action, next_latent, rew, done = batch

        log_pi, mu, sigma, rew_hat, done_hat = self(latent, action)

        gmm_loss = compute_gmm_loss(
            next_latent,
            log_pi,
            mu,
            sigma,
        )

        if self.predicting_termination:
            bce_loss = F.binary_cross_entropy_with_logits(done_hat, done)
        else:
            bce_loss = 0

        if self.predicting_reward:
            mse_loss = F.mse_loss(rew_hat, rew)
        else:
            mse_loss = 0

        return gmm_loss + bce_loss + mse_loss


def compute_gmm_loss(latent, log_pi, mu, sigma):
    """
    Args:
        latent  seq_len x batch_size x latent_size
        log_pi  seq_len x batch_size x num_gaussians
        mu      seq_len x batch_size x num_gaussians x latent_size
        sigma   seq_len x batch_size x num_gaussians x latent_size
    Returns:
        loss    1
    """
    latent = latent.unsqueeze(-2)

    latent_log_prob = Normal(mu, sigma).log_prob(latent)

    loss = -torch.logsumexp(log_pi + latent_log_prob.sum(dim=-1))

    return loss
