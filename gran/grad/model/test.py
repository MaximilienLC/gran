import torch.nn as nn
import torch


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        cfg = cfg

        self.k = nn.Linear(cfg.embd_dim, cfg.embd_dim)
        self.q = nn.Linear(cfg.embd_dim, cfg.embd_dim)
        self.v = nn.Linear(cfg.embd_dim, cfg.embd_dim)
        self.proj = nn.Linear(cfg.embd_dim, cfg.embd_dim)

        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.resid_dropout = nn.Dropout(cfg.resid_dropout)

        self.mask = torch.tril(torch.ones(cfg.seq_len, cfg.seq_len)).view(
            1, 1, cfg.seq_len, cfg.seq_len
        )

    def forward(self, x):

        batch_size, seq_len, embd_dim = x.size()

        k = self.k(x)
        q = self.q(x)
        v = self.v(x)

        x = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        x = x.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )
        x = F.softmax(x, dim=-1)
        x = self.attn_dropout(x)

        x = torch.matmul(x, v)
        x = self.resid_dropout(self.proj(x))

        return x
