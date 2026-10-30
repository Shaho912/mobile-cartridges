"""Per-group fake quantization: quantize to int4/int8, dequantize back to the model dtype.

Applied to every ``nn.Linear`` and ``nn.Embedding`` weight, including the input embedding.
When the embedding is tied to ``lm_head`` both modules share one ``Parameter``; it is
quantized once and stays tied. Norm weights and biases are left in the model dtype and
are counted at 16 bits in the analytic size.

Scheme: asymmetric min/max per contiguous group of ``group_size`` elements along the last
(input) dimension, with the zero point forced into range so 0.0 is exactly representable.
Scale and zero point are rounded to bf16 before use, matching what a deployment would store.
"""

from __future__ import annotations

import torch
from torch import nn

QUANT_BITS = {"bf16": 16, "int8": 8, "int4": 4}
SCALE_BYTES = 2
ZERO_BYTES = 2
_ROW_CHUNK = 8192


def quantizable_weights(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Unique Linear/Embedding weight parameters (tied weights appear once)."""
    seen: set[int] = set()
    out: list[tuple[str, nn.Parameter]] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            weight = module.weight
            if id(weight) in seen:
                continue
            seen.add(id(weight))
            out.append((f"{name}.weight", weight))
    return out


def _quant_dequant_rows(rows: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    n_rows, n_cols = rows.shape
    g = rows.float().reshape(n_rows, n_cols // group_size, group_size)
    qmax = 2**bits - 1
    lo = g.amin(-1, keepdim=True).clamp(max=0.0)
    hi = g.amax(-1, keepdim=True).clamp(min=0.0)
    scale = ((hi - lo) / qmax).clamp_min(1e-12).to(torch.bfloat16).float()
    zero = torch.clamp(torch.round(-lo / scale), 0, qmax)
    q = torch.clamp(torch.round(g / scale) + zero, 0, qmax)
    return ((q - zero) * scale).reshape(n_rows, n_cols)


def quant_dequant(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Return a dequantized copy of a 2-D weight in its original dtype."""
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2-D weight, got shape {tuple(weight.shape)}.")
    if weight.shape[1] % group_size:
        raise ValueError(
            f"Last dim {weight.shape[1]} is not divisible by group_size {group_size}."
        )
    out = torch.empty_like(weight)
    for start in range(0, weight.shape[0], _ROW_CHUNK):
        chunk = weight[start : start + _ROW_CHUNK]
        out[start : start + _ROW_CHUNK] = _quant_dequant_rows(chunk, bits, group_size).to(
            weight.dtype
        )
    return out


def analytic_weight_bytes(model: nn.Module, quant: str, group_size: int) -> dict:
    """Deployed weight size: quantized params * bits / 8 + per-group scale/zero + 16-bit rest.

    Works on a meta-device model, so no weights need to be downloaded.
    """
    bits = QUANT_BITS[quant]
    total_params = sum(p.numel() for p in model.parameters())
    qweights = quantizable_weights(model)
    q_params = sum(p.numel() for _, p in qweights)
    embed_params = sum(
        p.numel() for name, p in qweights if isinstance(_module_for(model, name), nn.Embedding)
    )
    if bits == 16:
        weight_bytes = total_params * 2
        overhead = 0
        num_groups = 0
    else:
        num_groups = q_params // group_size
        overhead = num_groups * (SCALE_BYTES + ZERO_BYTES)
        weight_bytes = q_params * bits / 8 + overhead + (total_params - q_params) * 2
    return {
        "quant": quant,
        "bits": bits,
        "group_size": group_size if bits < 16 else None,
        "total_params": total_params,
        "quantized_params": q_params if bits < 16 else 0,
        "embedding_params": embed_params,
        "num_groups": num_groups,
        "scale_zero_bytes": overhead,
        "weight_bytes": int(weight_bytes),
    }


def _module_for(model: nn.Module, param_name: str) -> nn.Module:
    return model.get_submodule(param_name.removesuffix(".weight"))


@torch.no_grad()
def fake_quantize_(model: nn.Module, quant: str, group_size: int) -> dict:
    """Quantize-dequantize all Linear/Embedding weights in place. Returns a size/error report."""
    report = analytic_weight_bytes(model, quant, group_size)
    bits = QUANT_BITS[quant]
    if bits == 16:
        report["weighted_rel_error"] = 0.0
        return report
    err_sq = 0.0
    norm_sq = 0.0
    weights = quantizable_weights(model)
    for _, weight in weights:
        deq = quant_dequant(weight.data, bits, group_size)
        err_sq += float((deq.float() - weight.data.float()).pow(2).sum())
        norm_sq += float(weight.data.float().pow(2).sum())
        weight.data.copy_(deq)
    report["weighted_rel_error"] = (err_sq / norm_sq) ** 0.5 if norm_sq else 0.0
    report["quantized_tensors"] = len(weights)
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    report["embedding_tied"] = output_emb is not None and output_emb.weight is input_emb.weight
    return report
