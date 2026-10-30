#!/usr/bin/env python3
"""Expand configs/size_sweep.yaml into a SLURM array manifest (one sweep_run.py call per line).

"max" slot runs are resolved here, per model x quantization x KV precision x patient:
the largest slot count with weights + cartridge KV <= budget, capped at the length of that
patient's record (prefix initialization cannot use more slots than the record has tokens).
Configurations whose weights alone exceed the budget get no max run.

Usage:
    python scripts/sweep_make_jobs.py --config configs/size_sweep.yaml
    sbatch --array=0-<N-1>%8 slurm/size_sweep_array.sbatch
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from cartridges.benchmarks.text_benchmark import context_system_prompt  # noqa: E402
from cartridges.sweep.budget import max_slots_under_budget  # noqa: E402
from cartridges.sweep.config import load_config  # noqa: E402
from cartridges.sweep.longhealth import load_benchmark, patient_record, patient_text  # noqa: E402
from cartridges.sweep.models import PromptAdapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/size_sweep.yaml")
    parser.add_argument("--models", nargs="*", default=None, help="Restrict to these model ids.")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    bench = load_benchmark(cfg["benchmark_json"])
    texts = {p: patient_text(patient_record(bench, p)) for p in cfg["patients"]}
    gs = cfg["quant_group_size"]
    jobs: list[dict] = []
    skipped: list[str] = []

    for spec in cfg["models"]:
        model_id = spec["id"]
        if args.models and model_id not in args.models:
            continue
        adapter = PromptAdapter(AutoTokenizer.from_pretrained(model_id), model_id)
        doc_tokens = {
            p: adapter.num_text_tokens(adapter.init_text(context_system_prompt(t)))
            for p, t in texts.items()
        }
        kv_eval = spec["kv_eval"]
        for quant in spec["quants"]:
            for p in cfg["patients"]:
                common = {"model": model_id, "quant": quant, "patient": p}
                for mode in spec["modes"]:
                    jobs.append({**common, "mode": mode})
                for slots in spec["slots"]:
                    jobs.append({**common, "mode": "cartridge", "slots": slots,
                                 "slot_tag": f"s{slots}", "kv_eval": kv_eval})
                for kv in spec["max_slots_kv"]:
                    m = max_slots_under_budget(model_id, quant, gs, kv, cfg["budget_mb"],
                                               doc_tokens[p])
                    if m["weights_over_budget"]:
                        if p == cfg["patients"][0]:
                            skipped.append(f"{model_id} {quant}: weights "
                                           f"{m['weight_bytes'] / 1e6:.0f} MB >= budget, "
                                           f"no max-slot run ({kv} KV)")
                        continue
                    jobs.append({**common, "mode": "cartridge", "slots": m["slots"],
                                 "slot_tag": f"max_{kv}kv", "kv_eval": kv_eval,
                                 "max_binding": m["binding"]})

    jobs_dir = cfg["output_root"] / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (cfg["output_root"] / "slurm").mkdir(parents=True, exist_ok=True)
    lines = []
    for j in jobs:
        parts = ["--model", j["model"], "--quant", j["quant"], "--mode", j["mode"],
                 "--patient", str(j["patient"])]
        if j["mode"] == "cartridge":
            parts += ["--slots", str(j["slots"]), "--slot-tag", j["slot_tag"],
                      "--kv-eval", *j["kv_eval"]]
        lines.append(" ".join(parts))
    (jobs_dir / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (jobs_dir / "manifest.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")

    for note in skipped:
        print(f"note: {note}")
    for j in jobs:
        if j.get("slot_tag", "").startswith("max"):
            print(f"max slots: {j['model']} {j['quant']} {j['slot_tag']} p{j['patient']:02d} -> "
                  f"{j['slots']} ({j['max_binding']})")
    print(f"\n{len(jobs)} jobs -> {jobs_dir / 'manifest.txt'}")
    print(f"sbatch --array=0-{len(jobs) - 1}%{args.concurrency} slurm/size_sweep_array.sbatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
