from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
from typeguard import typechecked


@typechecked
class FCVAE(pl.LightningModule):
    def __init__(self, x_size: int, latent_size: int) -> None:

        super().__init__()

        self.latent_size = latent_size

        self.encoder = _Encoder(x_size, latent_size)
        self.decoder = _Decoder(x_size, latent_size)

    def forward(
        self, x: TensorType["batch_size", "x_size"]
    ) -> Tuple[
        TensorType["batch_size", "x_size"],
        TensorType["batch_size", "latent_size"],
        TensorType["batch_size", "latent_size"],
    ]:

        mu, log_sigma = self.encoder(x)

        sigma = torch.exp(log_sigma)
        epsilon = torch.randn(self.latent_size, device=self.device)

        z = mu + sigma * epsilon

        recon_x = self.decoder(z)

        return recon_x, mu, log_sigma

    def training_step(
        self,
        batch: TensorType["batch_size", "x_size"],
        _: int,
    ) -> TensorType[1]:

        x = batch
        recon_x, mu, log_sigma = self(x)
        loss = compute_loss(x, recon_x, mu, log_sigma)

        return loss

    def validation_step(
        self,
        batch: TensorType["batch_size", "x_size"],
        _: int,
    ) -> None:

        x = batch
        recon_x, mu, log_sigma = self(x)
        loss = compute_loss(x, recon_x, mu, log_sigma)

        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())


def compute_loss(
    x: TensorType["batch_size", "x_size"],
    recon_x: TensorType["batch_size", "x_size"],
    mu: TensorType["batch_size", "latent_size"],
    log_sigma: TensorType["batch_size", "latent_size"],
) -> TensorType[1]:

    recon_loss = F.mse_loss(recon_x, x, reduction="sum")

    kl_loss = -0.5 * torch.sum(
        1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp()
    )

    return recon_loss + kl_loss


@typechecked
class _Encoder(nn.Module):
    def __init__(self, x_size: int, latent_size: int) -> None:

        super().__init__()

        self.latent_size = latent_size

        self.model = nn.Sequential(
            nn.Linear(x_size, latent_size * 2),
            nn.ReLU(),
        )

    def forward(
        self, x: TensorType["batch_size", "x_size"]
    ) -> Tuple[
        TensorType["batch_size", "latent_size"],
        TensorType["batch_size", "latent_size"],
    ]:

        mu_log_sigma = self.model(x)
        mu, log_sigma = mu_log_sigma.split(self.latent_size, dim=-1)

        return mu, log_sigma


@typechecked
class _Decoder(nn.Module):
    def __init__(self, x_size: int, latent_size: int) -> None:

        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_size, x_size),
            nn.Tanh(),
        )

    def forward(
        self, z: TensorType["batch_size", "latent_size"]
    ) -> TensorType["batch_size", "x_size"]:

        recon_x = self.model(z)

        return recon_x
