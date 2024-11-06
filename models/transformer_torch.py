"""Transformer model in plain PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TransformerConfig:
    """Hyperparameters used in the Transformer architectures."""
    def __init__(
        self,
        output_size: int,
        embedding_dim: int = 64,
        num_layers: int = 5,
        num_heads: int = 8,
        num_hiddens_per_head: Optional[int] = None,
        dropout_prob: float = 0.1,
        use_embeddings: bool = True,
        share_embeddings: bool = False,
        attention_window: Optional[int] = None,
        max_time: int = 10_000,
        widening_factor: int = 4,
        causal_masking: bool = False
    ):
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_hiddens_per_head = num_hiddens_per_head or (embedding_dim // num_heads)
        self.dropout_prob = dropout_prob
        self.use_embeddings = use_embeddings
        self.share_embeddings = share_embeddings
        self.attention_window = attention_window
        self.max_time = max_time
        self.widening_factor = widening_factor
        self.causal_masking = causal_masking


class MultiHeadDotProductAttention(nn.Module):
    """Multi-head dot-product attention."""
    def __init__(self, num_heads: int, num_hiddens_per_head: int, dropout_prob: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_hiddens_per_head = num_hiddens_per_head
        self.scaling_factor = 1.0 / (num_hiddens_per_head ** 0.5)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, _ = q.size()
        q = q.view(batch_size, seq_length, self.num_heads, self.num_hiddens_per_head).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.num_hiddens_per_head).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.num_hiddens_per_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling_factor
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        return output


class TransformerEncoderLayer(nn.Module):
    """A single layer of the Transformer encoder."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attention = MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            num_hiddens_per_head=config.num_hiddens_per_head,
            dropout_prob=config.dropout_prob
        )
        self.linear1 = nn.Linear(config.embedding_dim, config.embedding_dim * config.widening_factor)
        self.linear2 = nn.Linear(config.embedding_dim * config.widening_factor, config.embedding_dim)
        self.layer_norm1 = nn.LayerNorm(config.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention block
        attn_output = self.self_attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))

        # Feed-forward block
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder module."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_layers)])
        self.embedding = nn.Embedding(config.output_size, config.embedding_dim) if config.use_embeddings else None
        self.positional_encoding = self._generate_positional_encoding(config.max_time, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding:
            x = self.embedding(x) * (self.config.embedding_dim ** 0.5)

        x += self.positional_encoding[:x.size(1), :].unsqueeze(0).to(x.device)
        x = self.dropout(x)

        mask = None
        if self.config.causal_masking:
            mask = torch.tril(torch.ones((x.size(1), x.size(1)), dtype=torch.uint8, device=x.device)).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, mask)
        return x

    def _generate_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class Transformer(nn.Module):
    """Complete Transformer model with encoder and decoder."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.output_layer = nn.Linear(config.embedding_dim, config.output_size)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        encoded_output = self.encoder(src)
        output = self.output_layer(encoded_output)
        return output
