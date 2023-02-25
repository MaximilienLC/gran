from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DOTConfig:
    num_layers: int
    num_heads: int
    embd_dim: int

    dropout: float


class DecoderOnlyTransformer(nn.Module):
    """
    .
    """

    def __init__(self, cfg: DOTConfig) -> None:
        """
        .
        """
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


class DOTBlock(nn.Module):
    """
    .
    """

    def __init__(self, cfg):
        """
        .
        """
        super().__init__()

        self.ln1 = nn.LayerNorm(cfg.embd_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.embd_dim)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = self.attn(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x
        return x


class CausalSelfAttention(nn.Module):
    """
    Causal Self Attention
    """

    def __init__(self, config: DecoderOnlyTransformer) -> None:
        """
        .
        """
        super().__init__()

        self.key_query_value = nn.Linear(
            config.embd_dim, config.embd_dim * 3, bias=config.bias
        )

        self.proj = nn.Linear(
            config.embd_dim, config.embd_dim, bias=config.bias
        )

        self.resid_dropout = nn.Dropout(config.dropout)

        self.num_heads = config.num_heads

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        """
        .
        """
        BS, SL, ED = x.shape
        NH = self.num_heads
        HS = ED // NH

        # (BS, SL, ED) -> (BS, SL, ED*3) -> 3*(BS, SL, ED)
        k, q, v = self.key_query_value(x).split(ED, dim=2)

        # (BS, SL, ED) -> (BS, SL, NH, HS) -> (BS, NH, SL, HS)
        k = k.view(BS, SL, NH, HS).transpose(1, 2)
        q = q.view(BS, SL, NH, HS).transpose(1, 2)
        v = v.view(BS, SL, NH, HS).transpose(1, 2)

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )

        # (BS, NH, SL, HS) -> (BS, SL, NH, HS) -> (BS, SL, ED)
        x = x.transpose(1, 2).contiguous().view(BS, SL, ED)

        x = self.resid_dropout(self.proj(x))

        return x


class MLP(nn.Module):
    """
    Multi Layer Perceptron
    """

    def __init__(self, config: DecoderOnlyTransformer):
        """
        .
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(config.embd_dim, 4 * config.embd_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.embd_dim, config.embd_dim, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.model(x)
