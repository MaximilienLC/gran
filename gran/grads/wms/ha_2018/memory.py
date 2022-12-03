from typing import Tuple

import pytorch_lightning as pl
import torch
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
from typeguard import typechecked


@typechecked
class MDNRNN(pl.LightningModule):
    def __init__(
        self,
        latent_size: int,
        nb_actions: int,
        hidden_size: int,
        nb_gaussians: int,
        predicting_reward: bool = True,
        predicting_termination: bool = True,
    ) -> None:

        super().__init__()

        self.latent_size = latent_size
        self.nb_actions = nb_actions
        self.hidden_size = hidden_size
        self.nb_gaussians = nb_gaussians

        self.predicting_reward = predicting_reward
        self.predicting_termination = predicting_termination

        self.lstm = nn.LSTM(latent_size + nb_actions, hidden_size)
        self.fc = nn.Linear(
            hidden_size, (1 + 2 * latent_size) * nb_gaussians + 2
        )

    def forward(
        self,
        latent: TensorType["seq_len", "batch_size", "latent_size"],
        action: TensorType["seq_len", "batch_size", "nb_actions"],
    ) -> Tuple[
        TensorType["seq_len", "batch_size", "nb_gaussians"],
        TensorType["seq_len", "batch_size", "nb_gaussians", "latent_size"],
        TensorType["seq_len", "batch_size", "nb_gaussians", "latent_size"],
        TensorType["seq_len", "batch_size"],
        TensorType["seq_len", "batch_size"],
    ]:

        seq_len, batch_size = latent.size(0), latent.size(1)

        latent_and_action = torch.cat([latent, action], dim=-1)

        lstm_out, _ = self.lstm(latent_and_action)
        fc_out = self.linear(lstm_out)

        raw_pi, mu, log_sigma, rew_hat, done_hat = fc_out.split(
            [
                self.nb_gaussians,
                self.nb_gaussians * self.latent_size,
                self.nb_gaussians * self.latent_size,
                1,
                1,
            ],
            dim=-1,
        )

        log_pi = F.log_softmax(
            raw_pi.view(seq_len, batch_size, self.nb_gaussians), dim=-1
        )

        mu = mu.view(seq_len, batch_size, self.nb_gaussians, self.latent_size)

        sigma = torch.exp(
            log_sigma.view(
                seq_len, batch_size, self.nb_gaussians, self.latent_size
            )
        )

        return log_pi, mu, sigma, rew_hat, done_hat

    def training_step(self, batch: torch.Tensor, _: int) -> torch.Tensor:

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

    def validation_step(self, batch: torch.Tensor, _: int) -> torch.Tensor:

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

        loss = gmm_loss + bce_loss + mse_loss

        self.log("val/loss", loss, prog_bar=True)


def compute_gmm_loss(
    latent: TensorType["seq_len", "batch_size", "latent_size"],
    log_pi: TensorType["seq_len", "batch_size", "nb_gaussians"],
    mu: TensorType["seq_len", "batch_size", "nb_gaussians", "latent_size"],
    sigma: TensorType["seq_len", "batch_size", "nb_gaussians", "latent_size"],
) -> TensorType[1]:

    latent = latent.unsqueeze(-2)

    latent_log_prob = Normal(mu, sigma).log_prob(latent)

    loss = -torch.logsumexp(log_pi + latent_log_prob.sum(dim=-1))

    return loss
