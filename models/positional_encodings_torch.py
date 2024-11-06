import enum
import math
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn


class PositionalEncodings(enum.Enum):
    """Enum for all the positional encodings implemented."""
    NONE = 0
    SIN_COS = 1
    ALIBI = 2
    RELATIVE = 3
    ROTARY = 4


class SinCosParams:
    """Parameters for the classical sin/cos positional encoding."""

    def __init__(self, max_time: int = 10_000):
        self.max_time = max_time


RotaryParams = SinCosParams
RelativeParams = SinCosParams

POS_ENC_TABLE = {
    'NONE': PositionalEncodings.NONE,
    'SIN_COS': PositionalEncodings.SIN_COS,
    'ALIBI': PositionalEncodings.ALIBI,
    'RELATIVE': PositionalEncodings.RELATIVE,
    'ROTARY': PositionalEncodings.ROTARY,
}

POS_ENC_PARAMS_TABLE = {
    'NONE': SinCosParams,
    'SIN_COS': SinCosParams,
    'ALIBI': SinCosParams,
    'RELATIVE': RelativeParams,
    'ROTARY': RotaryParams,
}


def sinusoid_position_encoding(
        sequence_length: int,
        hidden_size: int,
        memory_length: int = 0,
        max_timescale: float = 1e4,
        min_timescale: float = 2.0,
        clamp_length: int = 0,
        causal: bool = False
):
    """Creates sinusoidal encodings."""
    freqs = np.arange(0, hidden_size, min_timescale)
    inv_freq = max_timescale ** (-freqs / hidden_size)

    if causal:
        pos_seq = np.arange(sequence_length + memory_length, 0, -1.0)
    else:
        pos_seq = np.arange(sequence_length + memory_length, -sequence_length, -1.0)

    if clamp_length:
        pos_seq = np.clip(pos_seq, a_min=-clamp_length, a_max=clamp_length)

    sinusoid_inp = np.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = np.concatenate([np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1)

    return torch.tensor(pos_emb, dtype=torch.float32)


def _rel_shift_inner(logits: torch.Tensor, attention_length: int) -> torch.Tensor:
    """Shifts the relative logits."""
    tq, total_len = logits.shape
    assert total_len == tq + attention_length
    logits = logits.reshape(total_len, tq)
    logits = logits[1:]  # logits[1:]
    logits = logits.reshape(tq, total_len - 1)
    return logits[:, :attention_length]


def _rel_shift_causal(logits: torch.Tensor) -> torch.Tensor:
    """Shifts the relative logits, assuming causal attention."""
    t1, t2 = logits.shape
    to_pad = torch.zeros_like(logits[..., :1])
    x = torch.cat((to_pad, logits), dim=-1)
    x = x.reshape(t2 + 1, t1)
    x = x[1:]  # Remove extra time dimension
    return x.reshape(t1, t2)


def relative_shift(logits: torch.Tensor, attention_length: int, causal: bool = False) -> torch.Tensor:
    if causal:
        return _rel_shift_causal(logits)
    else:
        return _rel_shift_inner(logits, attention_length)


def apply_rotary_encoding(x: torch.Tensor, position: torch.Tensor, max_time: int = 10_000) -> torch.Tensor:
    """Applies RoPE positional encodings for the input."""
    freq_seq = torch.arange(x.shape[-1] // 2, dtype=torch.float32)
    inv_freq = max_time ** -(freq_seq / (x.shape[-1] // 2))
    inv_freq = inv_freq.repeat(2)

    t = position[:, :, None, None] * inv_freq[None, None, None, :]
    x_rot = torch.einsum('bthd,dD->bthD', x, _rope_kernel(x.shape[-1], x.dtype))
    return x * torch.cos(t).type_as(x) + torch.sin(t).type_as(x) * x_rot


def _rope_kernel(n: int, dtype: torch.dtype) -> torch.Tensor:
    """Reorders the embedding dimension of an array, to make rotation easier."""
    assert n % 2 == 0, n
    kernel = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if i % 2 == 0:
            kernel[i, i + 1] = 1
        else:
            kernel[i, i - 1] = -1
    return torch.tensor(kernel, dtype=dtype)


def compute_attention_with_relative_encodings(
        queries: torch.Tensor,
        keys: torch.Tensor,
        max_time: int = 10_000,
        causal: bool = False
) -> torch.Tensor:
    """Returns attention with relative positional encodings."""
    batch_size, k_seq_len, num_heads, num_hiddens = keys.shape
    hiddens = num_hiddens * num_heads

    # Content logits computation
    content_bias = torch.randn((num_heads, num_hiddens), std=0.02, device=queries.device)
    content_logits = torch.einsum('bthd,bThd->bhtT', queries + content_bias, keys)

    positional_encodings = sinusoid_position_encoding(
        sequence_length=k_seq_len,
        hidden_size=hiddens,
        memory_length=0,
        max_timescale=max_time,
        min_timescale=2,
        clamp_length=0,
        causal=causal,
    ).to(queries.device)

    relative_keys = nn.Linear(hiddens, hiddens, bias=False)(positional_encodings)
    relative_keys = relative_keys.view(batch_size, k_seq_len, num_heads, num_hiddens)

    # Relative logits computation
    relative_bias = torch.randn((num_heads, num_hiddens), std=0.02, device=queries.device)
    relative_logits = torch.einsum('bthd,bThd->bhtT', queries + relative_bias, relative_keys)
    relative_logits = relative_shift(relative_logits, attention_length=content_logits.shape[-1], causal=causal)

    assert content_logits.shape == relative_logits.shape
    return content_logits + relative_logits


def _get_alibi_slopes(num_heads: int) -> list[float]:
    """Returns the slopes for the different attention heads."""

    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        return (get_slopes_power_of_2(closest_power_of_2) +
                _get_alibi_slopes(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2])


def compute_alibi_encodings_biases(
        attention_shape: tuple[int, ...]
) -> torch.Tensor:
    """Returns the biases following the ALiBi method."""
    num_heads, q_seq_len, k_seq_len = attention_shape

    alibi = np.zeros((q_seq_len, k_seq_len))
    alibi -= sum(np.tri(*alibi.shape, k=-i) for i in range(1, q_seq_len))
    alibi -= sum(np.tri(*alibi.T.shape, k=-i).T for i in range(1, k_seq_len))
    alibi += 0.5 * np.tri(*alibi.shape, k=-1)

    alibi_slopes = torch.tensor(_get_alibi_slopes(num_heads), dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
    return torch.tensor(alibi, dtype=torch.float32) * alibi_slopes
