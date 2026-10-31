#!/usr/bin/env python3
"""Per-patient shared inputs for the size sweep: eval sets and bootstrap training questions.

Eval sets come straight from longhealth_benchmark_v5.json (all 20 questions per patient,
correct letter included). Bootstrap questions are generated ONCE per patient by the vLLM
server and reused by every model and quantization, so models differ only in their own
teacher logprobs, never in the training questions.

Usage (vLLM server running with the bootstrap model):
    python scripts/sweep_bootstrap.py --config configs/size_sweep.yaml
    python scripts/sweep_bootstrap.py --config configs/size_sweep.yaml --eval-only
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.sweep.config import bootstrap_path, load_config  # noqa: E402
from cartridges.sweep.longhealth import (  # noqa: E402
    eval_items,
    load_benchmark,
    patient_record,
    patient_text,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/size_sweep.yaml")
    parser.add_argument("--eval-only", action="store_true", help="Write eval sets; skip vLLM.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing bootstrap files.")
    parser.add_argument("--base-url", help="vLLM server URL; overrides bootstrap.base_url.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bench = load_benchmark(cfg["benchmark_json"])
    eval_dir = cfg["output_root"] / "eval_sets"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for p in cfg["patients"]:
        record = patient_record(bench, p)
        items = eval_items(record, p, cfg["questions_per_patient"])
        (eval_dir / f"lh_p{p:02d}.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
        no_letter = [i["id"] for i in items if i["correct_letter"] is None]
        print(f"[lh_p{p:02d}] {len(items)} eval questions"
              + (f"; correct text matches no single option: {no_letter}" if no_letter else ""))

    if args.eval_only:
        return 0

    from cartridges.benchmarks.text_benchmark import generate_bootstrap_questions
    from cartridges.config import DEFAULT_MATRIX

    boot = cfg["bootstrap"]
    if DEFAULT_MATRIX.model_id != boot["vllm_model"]:
        raise SystemExit(
            f"CARTRIDGES_MODEL_ID={DEFAULT_MATRIX.model_id!r} but config bootstrap.vllm_model="
            f"{boot['vllm_model']!r}. Export CARTRIDGES_MODEL_ID to match the vLLM server."
        )
    for p in cfg["patients"]:
        out = bootstrap_path(cfg, p)
        if out.exists() and not args.force:
            print(f"[lh_p{p:02d}] bootstrap exists, skipping: {out}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        text = patient_text(patient_record(bench, p))
        examples = generate_bootstrap_questions(
            corpus_text=text,
            eval_spec=[],
            output_path=out.with_suffix(".questions.txt"),
            base_url=args.base_url or boot["base_url"],
            api_key=boot["api_key"],
            num_questions=boot["count"],
            spread=boot.get("spread", False),
        )
        out.write_text(json.dumps({
            "patient": p,
            "vllm_model": boot["vllm_model"],
            "num_questions": len(examples),
            "examples": examples,
        }, indent=2), encoding="utf-8")
        print(f"[lh_p{p:02d}] {len(examples)} bootstrap Q&A -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
