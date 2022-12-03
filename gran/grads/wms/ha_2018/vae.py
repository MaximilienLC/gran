from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
from typeguard import typechecked


@typechecked
class VAE(pl.LightningModule):
    def __init__(self, latent_size: int) -> None:

        super().__init__()

        self.latent_size = latent_size

        self.encoder = _Encoder(latent_size)
        self.decoder = _Decoder(latent_size)

    def forward(
        self, x: TensorType["batch_size", "nb_channels", "height", "width"]
    ) -> Tuple[
        TensorType["batch_size", "nb_channels", "height", "width"],
        TensorType["batch_size", "latent_size"],
        TensorType["batch_size", "latent_size"],
    ]:

        mu, log_sigma = self.encoder(x)

        sigma = torch.exp(log_sigma)
        epsilon = torch.randn(self.latent_size)

        z = mu + sigma * epsilon

        recon_x = self.decoder(z)

        return recon_x, mu, log_sigma

    def training_step(
        self,
        batch: TensorType["batch_size", "nb_channels", "height", "width"],
        _: int,
    ) -> TensorType[1]:

        x = batch
        recon_x, mu, log_sigma = self(x)
        loss = compute_loss(x, recon_x, mu, log_sigma)

        return loss

    def validation_step(
        self,
        batch: TensorType["batch_size", "nb_channels", "height", "width"],
        _: int,
    ) -> None:

        x = batch
        recon_x, mu, log_sigma = self(x)
        loss = compute_loss(x, recon_x, mu, log_sigma)

        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())


def compute_loss(
    x: TensorType["batch_size", "nb_channels", "height", "width"],
    recon_x: TensorType["batch_size", "nb_channels", "height", "width"],
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
    def __init__(self, latent_size: int) -> None:

        super().__init__()

        self.latent_size = latent_size

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, latent_size * 2),
        )

    def forward(
        self, x: TensorType["batch_size", "nb_channels", "height", "width"]
    ) -> Tuple[
        TensorType["batch_size", "latent_size"],
        TensorType["batch_size", "latent_size"],
    ]:

        mu_log_sigma = self.model(x)
        mu, log_sigma = mu_log_sigma.split(self.latent_size, dim=-1)

        return mu, log_sigma


@typechecked
class _Decoder(nn.Module):
    def __init__(self, latent_size: int) -> None:

        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_size, 1024),
            nn.ReLU(),
            nn.Unflatten(),
            nn.ConvTranspose2d(1024, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2),
            nn.Sigmoid(),
        )

    def forward(
        self, z: TensorType["batch_size", "latent_size"]
    ) -> TensorType["batch_size", "nb_channels", "height", "width"]:

        recon_x = self.model(z)

        return recon_x
