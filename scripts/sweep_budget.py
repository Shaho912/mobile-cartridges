#!/usr/bin/env python3
"""Budget calculator: weight MB per quantization + cartridge MB per slot count.

Reads num_hidden_layers / num_key_value_heads / head_dim from each model's config.json and
counts parameters on a meta-device model (no weight download). Flags every configuration
over the budget and prints the max slot count that fits. Totals are lower bounds
(activations excluded).

Usage:
    python scripts/sweep_budget.py --config configs/size_sweep.yaml
    python scripts/sweep_budget.py --config configs/size_sweep.yaml --doc-tokens   # cap by patient length
    python scripts/sweep_budget.py --config configs/size_sweep.yaml --bnb          # NF4 footprint, needs GPU
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.sweep.budget import (  # noqa: E402
    MB,
    budget_row,
    max_slots_under_budget,
    per_slot_bytes,
    weight_report,
)
from cartridges.sweep.config import load_config  # noqa: E402
from cartridges.sweep.kv_quant import KV_PRECISIONS  # noqa: E402
from cartridges.sweep.models import attention_info  # noqa: E402

SLOT_GRID = [128, 256, 512, 1024]


def _doc_tokens(cfg: dict, model_id: str) -> dict[int, int]:
    from transformers import AutoTokenizer

    from cartridges.benchmarks.text_benchmark import context_system_prompt
    from cartridges.sweep.longhealth import load_benchmark, patient_record, patient_text

    tok = AutoTokenizer.from_pretrained(model_id)
    bench = load_benchmark(cfg["benchmark_json"])
    return {
        p: len(tok.encode(context_system_prompt(patient_text(patient_record(bench, p))),
                          add_special_tokens=False))
        for p in cfg["patients"]
    }


def _bnb_footprint(model_id: str) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        return {"status": f"unavailable: {exc}"}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map={"": 0},
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        return {"status": f"unavailable: {exc}"}
    emb = model.get_input_embeddings().weight
    out = {
        "status": "ok",
        "nf4_footprint_mb": model.get_memory_footprint() / MB,
        "embedding_dtype": str(emb.dtype),
        "embedding_mb": emb.numel() * emb.element_size() / MB,
    }
    del model
    torch.cuda.empty_cache()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/size_sweep.yaml")
    parser.add_argument("--doc-tokens", action="store_true",
                        help="Tokenize each patient record and cap max slots at its length.")
    parser.add_argument("--bnb", action="store_true",
                        help="Also load each model in bitsandbytes NF4 and report its footprint.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    budget_mb = cfg["budget_mb"]
    gs = cfg["quant_group_size"]
    report: dict = {"budget_mb": budget_mb, "group_size": gs, "models": []}
    lines = [f"# Budget: {budget_mb} MB per chip (weights + cartridge KV, activations excluded)", ""]

    for spec in cfg["models"]:
        model_id = spec["id"]
        info = attention_info(model_id)
        per_slot = {kv: per_slot_bytes(model_id, kv) for kv in KV_PRECISIONS}
        docs = _doc_tokens(cfg, model_id) if args.doc_tokens else {}
        entry = {"model": model_id, "attention": info, "kv_bytes_per_slot": per_slot,
                 "doc_tokens": docs, "quants": []}

        lines += [
            f"## {model_id}",
            f"layers={info['num_hidden_layers']}  kv_heads={info['num_key_value_heads']}  "
            f"head_dim={info['head_dim']}  context={info['max_position_embeddings']}",
            "KV per slot: " + ", ".join(f"{kv} {b / 1e3:.1f} KB" for kv, b in per_slot.items()),
        ]
        if info["num_sliding_layers"]:
            lines.append(
                f"WARNING: {info['num_sliding_layers']} of {info['num_hidden_layers']} layers use a "
                f"{info['sliding_window']}-token sliding window; on those layers only the most "
                "recent slots are visible."
            )
        if docs:
            lines.append("Patient record tokens: " + ", ".join(f"p{p:02d}={n}" for p, n in docs.items()))
            over = {p: n for p, n in docs.items() if n > info["max_position_embeddings"]}
            if over:
                lines.append(f"DOES NOT FIT CONTEXT: {over}")
        lines += ["", "| quant | weight MB | embed params | " + " | ".join(
            f"{s} slots total MB" for s in SLOT_GRID) + " | max slots (bf16 KV) | max slots (int8 KV) |",
            "|---|---|---|" + "---|" * len(SLOT_GRID) + "---|---|"]

        for quant in ("bf16", "int8", "int4"):
            w = weight_report(model_id, quant, gs)
            doc_cap = min(docs.values()) if docs else None
            maxes = {kv: max_slots_under_budget(model_id, quant, gs, kv, budget_mb, doc_cap)
                     for kv in KV_PRECISIONS}
            rows = [budget_row(model_id, quant, gs, "bf16", s, budget_mb) for s in SLOT_GRID]
            entry["quants"].append({"weights": w, "slot_rows": rows, "max_slots": maxes})
            cells = [f"{r['total_mb']:.0f}{' OVER' if r['over_budget'] else ''}" for r in rows]
            max_cells = [
                "n/a (weights over budget)" if m["weights_over_budget"]
                else f"{m['slots']} ({m['binding']})"
                for m in maxes.values()
            ]
            lines.append(
                f"| {quant} | {w['weight_bytes'] / MB:.1f} | {w['embedding_params'] / 1e6:.1f}M | "
                + " | ".join(cells) + " | " + " | ".join(max_cells) + " |"
            )

        if args.bnb:
            entry["bnb_nf4"] = _bnb_footprint(model_id)
            lines.append(f"\nbitsandbytes NF4: {entry['bnb_nf4']}")
        lines.append("")
        report["models"].append(entry)

    out_dir = cfg["output_root"] / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "budget.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "budget.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out_dir / 'budget.json'} and budget.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
