"""Cartridge KV storage precision: byte accounting and int8 round-trip.

int8 storage is symmetric per (layer, kv_head, slot) vector over ``head_dim`` with one bf16
scale per vector. It is a post-training ablation: the cartridge is trained in bf16, then
stored as int8 and dequantized at load.
"""

from __future__ import annotations

import torch

from cartridges.core import TrainableKVCartridge

KV_PRECISIONS = ("bf16", "int8")
_SCALE_BYTES = 2


def kv_bytes_per_slot(num_layers: int, kv_heads: int, head_dim: int, precision: str) -> int:
    """Bytes for one slot across all layers, keys + values."""
    if precision == "bf16":
        per_vector = head_dim * 2
    elif precision == "int8":
        per_vector = head_dim + _SCALE_BYTES
    else:
        raise ValueError(f"Unknown KV precision {precision!r}.")
    return 2 * num_layers * kv_heads * per_vector


def _int8_roundtrip(t: torch.Tensor) -> torch.Tensor:
    tf = t.float()
    amax = tf.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / 127.0).to(torch.bfloat16).float()
    q = torch.clamp(torch.round(tf / scale), -127, 127)
    return (q * scale).to(t.dtype)


@torch.no_grad()
def with_kv_precision(cartridge: TrainableKVCartridge, precision: str) -> TrainableKVCartridge:
    """Return the cartridge as it would be after storage at ``precision`` and reload."""
    if precision == "bf16":
        return cartridge
    if precision != "int8":
        raise ValueError(f"Unknown KV precision {precision!r}.")
    keys, values = [], []
    for i in range(cartridge.num_layers):
        k, v = cartridge.layer(i)
        keys.append(_int8_roundtrip(k))
        values.append(_int8_roundtrip(v))
    return TrainableKVCartridge(
        keys=keys, values=values, num_frozen_tokens=cartridge.num_frozen_tokens
    )
