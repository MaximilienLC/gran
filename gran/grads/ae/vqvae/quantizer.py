import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQEmbedding(pl.LightningModule):
    def __init__(self, nb_embeddings, len_embedding):

        super().__init__()

        self.nb_embeddings = nb_embeddings
        self.len_embedding = len_embedding

        self.embedding = nn.Embedding(nb_embeddings, nb_embeddings)
        self.embedding.weight.data.uniform_(
            -1.0 / nb_embeddings, 1.0 / nb_embeddings
        )

    def decode(self, z_e_x):
        """
        Args:
            z_e_x batch_size x len_embedding

        """
        # dist_z_embd : (z - embd) ^ 2 = (z ^ 2) + (embd ^ 2) - (2 * embd * z)
        #          batch_size x nb_embeddings
        #                            batch_size x 1
        #                                        nb_embeddings
        #                                            batch_size x nb_embeddings
        dist_z_embd = (
            torch.sum(z_e_x**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_e_x, self.embedding.weight.t())
        )

        min_dist_indices = torch.argmin(dist_z_embd, dim=1).unsqueeze(1)

        z_q_x = self.embedding(min_dist_indices)

        z_q_x = z_e_x + (z_q_x - z_e_x).detach()

        return z_q_x
