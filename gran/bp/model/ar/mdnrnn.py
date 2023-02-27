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
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from gran.bp.model.base import BaseModel
from gran.util.misc import config


class MDNRNN(BaseModel):
    def __init__(self) -> None:
        """
        Config:
            num_features: Number of features
            num_actions: Number of allowed actions
            hidden_size: Number of hidden neurons
            num_gaussians: Number of gaussians
            predicting_reward: Whether to predict rewards
            predicting_termination: Whether to predict termination
        """
        super().__init__()

        self.num_features = config.predictor.num_features
        self.num_actions = config.predictor.num_actions
        self.hidden_size = config.predictor.hidden_size
        self.num_gaussians = config.predictor.num_gaussians
        self.predicting_reward = config.predictor.predicting_reward
        self.predicting_termination = config.predictor.predicting_termination

        self.lstm = nn.LSTM(
            input_size=self.num_features + self.num_actions,
            hidden_size=self.hidden_size,
            batch_first=True,
        )

        self.fc = nn.Linear(
            self.hidden_size,
            (1 + 2 * self.num_features) * self.num_gaussians + 2,
        )

    def reset(self):

        self.h = torch.zeros(1, 1, self.hidden_size)
        self.c = torch.zeros(1, 1, self.hidden_size)

    def forward(self, x, done=None):
        """
        Args:
            x             batch_size x seq_len x num_features + num_actions
            done          batch_size x seq_len
        Returns:
            fit:
                log_pi    batch_size x seq_len x num_gaussians
                mu        batch_size x seq_len x num_gaussians x num_features
                sigma     batch_size x seq_len x num_gaussians x num_features
                rew_hat   batch_size x seq_len
                done_hat  batch_size x seq_len
            pred:
                x_t+1     batch_size x 1 x num_features
                rew_hat   batch_size x 1
                done_hat  batch_size x 1
        """
        stage = "fit" if done == None else "pred"

        if stage == "fit":

            x = pack_padded_sequence(
                x,
                lengths=torch.argwhere(done)[:, 1].cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            self.reset()

        x, (self.h, self.c) = self.lstm(x, (self.h, self.c))

        if stage == "fit":
            x = x[0]
            print(x.shape)

        x = self.fc(x)

        raw_pi, mu, log_sigma, rew_hat, done_hat = x.split(
            [
                self.num_gaussians,
                self.num_gaussians * self.num_features,
                self.num_gaussians * self.num_features,
                1,
                1,
            ],
            dim=-1,
        )

        log_pi = F.log_softmax(raw_pi)

        mu = mu.view(x.size(0), self.num_gaussians, self.num_features)

        sigma = torch.exp(
            log_sigma.view(x.size(0), self.num_gaussians, self.num_features)
        )

        if stage == "fit":

            return log_pi, mu, sigma, rew_hat, done_hat

        else:  # stage == "pred":

            pi = torch.exp(log_pi).squeeze()

            gaussian_idx = torch.multinomial(pi, 1)

            mu, sigma = mu[:, :, gaussian_idx, :], sigma[:, :, gaussian_idx, :]

            return (
                mu + sigma * torch.randn_like(sigma),
                rew_hat,
                done_hat > 0.5,
            )

    def step(self, batch: torch.Tensor, stage: str):

        features_and_actions, next_features, rew, done = batch

        log_pi, mu, sigma, rew_hat, done_hat = self(features_and_actions, done)

        gmm_loss = compute_gmm_loss(
            next_features,
            log_pi,
            mu,
            sigma,
            lengths,
        )

        if self.predicting_termination:
            bce_loss = F.binary_cross_entropy_with_logits(done_hat, done)
        else:
            bce_loss = 0

        if self.predicting_reward:
            mse_loss = F.mse_loss(rew_hat, rew)
        else:
            mse_loss = 0

        self.log(f"{stage}/gmm_loss", gmm_loss)
        self.log(f"{stage}/bce_loss", bce_loss)
        self.log(f"{stage}/mse_loss", mse_loss)

        return gmm_loss + bce_loss + mse_loss


def compute_gmm_loss(next_features, log_pi, mu, sigma, lengths):
    """
    Args:
        next_features  batch_size x seq_len x num_features
        log_pi  batch_size x seq_len x num_gaussians
        mu      batch_size x seq_len x num_gaussians x num_features
        sigma   batch_size x seq_len x num_gaussians x num_features
    Returns:
        loss    1
    """

    next_features = pack_padded_sequence(
        next_features,
        lengths.squeeze().cpu(),
        batch_first=True,
        enforce_sorted=False,
    )[0]

    next_features = next_features.unsqueeze(-2)

    next_features_log_prob = Normal(mu, sigma).log_prob(next_features)

    loss = -torch.logsumexp(
        log_pi + next_features_log_prob.sum(dim=-1), dim=-1
    ).mean()

    return loss
