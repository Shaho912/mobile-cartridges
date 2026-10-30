"""Per-chip memory budget: weights (per quantization) + cartridge KV (per slot count).

Sizes are lower bounds: runtime activations, the query's own KV, and framework overhead are
excluded. 1 MB = 1e6 bytes.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from cartridges.sweep.fake_quant import analytic_weight_bytes
from cartridges.sweep.kv_quant import kv_bytes_per_slot
from cartridges.sweep.models import attention_info

MB = 1e6


@lru_cache(maxsize=None)
def _meta_model(model_id: str):
    config = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config)


def weight_report(model_id: str, quant: str, group_size: int) -> dict:
    return analytic_weight_bytes(_meta_model(model_id), quant, group_size)


def per_slot_bytes(model_id: str, kv_precision: str) -> int:
    info = attention_info(model_id)
    return kv_bytes_per_slot(
        info["num_hidden_layers"], info["num_key_value_heads"], info["head_dim"], kv_precision
    )


def max_slots_under_budget(
    model_id: str,
    quant: str,
    group_size: int,
    kv_precision: str,
    budget_mb: float,
    doc_tokens: int | None = None,
) -> dict:
    """Largest slot count with weights + KV <= budget, capped by document length if given."""
    weights = weight_report(model_id, quant, group_size)["weight_bytes"]
    slot_bytes = per_slot_bytes(model_id, kv_precision)
    budget_bytes = budget_mb * MB
    by_budget = int((budget_bytes - weights) // slot_bytes) if weights < budget_bytes else 0
    slots, binding = by_budget, "budget"
    if doc_tokens is not None and doc_tokens < by_budget:
        slots, binding = doc_tokens, "document_length"
    return {
        "slots": slots,
        "binding": binding,
        "slots_by_budget": by_budget,
        "weight_bytes": weights,
        "kv_bytes_per_slot": slot_bytes,
        "weights_over_budget": weights >= budget_bytes,
    }


def budget_row(
    model_id: str, quant: str, group_size: int, kv_precision: str, slots: int, budget_mb: float
) -> dict:
    weights = weight_report(model_id, quant, group_size)["weight_bytes"]
    kv = slots * per_slot_bytes(model_id, kv_precision)
    total = weights + kv
    return {
        "model": model_id,
        "quant": quant,
        "kv_precision": kv_precision,
        "slots": slots,
        "weight_mb": weights / MB,
        "cartridge_mb": kv / MB,
        "total_mb": total / MB,
        "over_budget": total > budget_mb * MB,
    }
