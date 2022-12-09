import torch.nn as nn

from gran.grads.vae.base import VAE


class MLPVAE(VAE):
    def __init__(
        self,
        x_size: int,
        hidden_size: int,
        latent_size: int,
        decoder_final_activation: nn.Module,
    ) -> None:

        encoder_model = nn.Sequential(
            nn.Linear(x_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size * 2),
            nn.ReLU(),
        )

        decoder_model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(latent_size, x_size),
            decoder_final_activation,
        )

        super().__init__(encoder_model, decoder_model)
