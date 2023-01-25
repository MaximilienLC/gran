from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:

    num_layers: int
    num_heads: int
    embd_dim: int

    dropout: float


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        cfg = cfg

        self.key = nn.Linear(cfg.embd_dim, cfg.embd_dim, bias=False)
        self.query = nn.Linear(cfg.embd_dim, cfg.embd_dim, bias=False)
        self.value = nn.Linear(cfg.embd_dim, cfg.embd_dim, bias=False)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.proj = nn.Linear(cfg.embd_dim, cfg.embd_dim, bias=False)

    def forward(self, x, kv_cache):

        # (BS, SL, ED)
        batch_size, seq_len, embd_dim = x.shape
        head_size = embd_dim // cfg.num_heads

        BS, SL, ED, NH, HS = (
            batch_size,
            seq_len,
            embd_dim,
            cfg.num_heads,
            head_size,
        )

        # (BS, SL, ED) -> (BS, SL, NH, HS) -> (BS, NH, SL, HS)
        q = self.query(x).view(BS, SL, NH, HS).transpose(1, 2)
        k = self.key(x).view(BS, SL, NH, HS).transpose(1, 2)
        v = self.value(x).view(BS, SL, NH, HS).transpose(1, 2)

        # (BS, NH, SL, HS) x (BS, NH, HS, SL) -> (BS, NH, SL, SL)
        x = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))

        x = x.masked_fill(self.mask[:, :, :SL, :SL] == 0, float("-inf"))
        x = F.softmax(x, dim=-1)
        x = self.attn_dropout(x)

        # (BS, NH, SL, SL) x (BS, NH, SL, HS) -> (BS, NH, SL, HS)
        x = torch.matmul(x, v)
        x = x.transpose(1, 2).contiguous().view(BS, SL, ED)

        x = self.resid_dropout(self.proj(x))

        return x


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(cfg.embd_dim, 4 * cfg.embd_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.embd_dim, cfg.embd_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.model(x)


class CausalSelfAttentionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.ln1 = nn.LayerNorm(cfg.embd_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.embd_dim)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = self.attn(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        cfg = cfg

        self.tok_embd = nn.Embedding(cfg.vocab_size, cfg.embd_dim)
        self.pos_embd = nn.Embedding(cfg.seq_len, cfg.embd_dim)
        self.blocks = nn.Sequential(
            *[CausalSelfAttentionBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.embd_dim)
        self.lm_head = nn.Linear(cfg.embd_dim, cfg.vocab_size, bias=False)

    def forward(self, idx):

        batch_size, seq_len = idx.size()

        tok_embd = self.tok_embd(idx)
        pos_embd = self.pos_embd(torch.arange(seq_len, device=idx.device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.lm_head(x)

        return x

    def generate(self, idx, max_new_tokens=10):

        batch_size, seq_len = idx.size()

        for _ in range(max_new_tokens):

            logits = self.forward(x)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            x = torch.cat(
                [
                    x,
                    torch.multinomial(probs, 1).detach(),
                ],
                dim=-1,
            )

        return x
