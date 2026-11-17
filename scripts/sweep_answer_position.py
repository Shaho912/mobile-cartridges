#!/usr/bin/env python3
"""Split eval questions by whether the answer is inside the untrained cartridge's prefix.

The untrained cartridge (tag <slots>_init) holds the first N tokens of system prompt + record
verbatim. For each LongHealth question this finds where the correct option's text first
appears in the record and labels it:

  prefix     inside those first N tokens (the untrained cartridge has it word for word)
  later      after them (only training can have put it in the cartridge)
  unlocated  neither an exact match nor a window holding >= 70% of its content words

Then per label: how many questions each run gets right, and trained-only / untrained-only
counts against the untrained cartridge. CPU only; reads existing result.json files.

Usage:
    python scripts/sweep_answer_position.py --config configs/size_sweep.yaml \\
        --quant-dir int4_g64 --tags s1024 s1024_mcq s1024_lr3e-4 s1024_mcq_lr3e-4
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from cartridges.benchmarks.text_benchmark import context_system_prompt  # noqa: E402
from cartridges.sweep.config import load_config, run_dir  # noqa: E402
from cartridges.sweep.longhealth import (  # noqa: E402
    eval_items,
    load_benchmark,
    patient_record,
    patient_text,
)

_STOP = {"the", "and", "was", "were", "with", "for", "from", "that", "this", "has", "had", "her",
         "his", "she", "patient", "which", "are", "not", "but", "after", "of", "in", "on", "to"}
_LABELS = ("prefix", "later", "unlocated")


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", text.lower())
            if len(w) >= 3 and w not in _STOP]


def locate(answer: str, record: str, window: int = 400, step: int = 100) -> tuple[int | None, str]:
    """First character offset of the answer in the record, and how it was found."""
    lowered = record.lower()
    pos = lowered.find(answer.lower())
    if pos >= 0:
        return pos, "exact"
    words = set(_words(answer))
    if not words:
        return None, "none"
    for start in range(0, max(1, len(lowered) - window + step), step):
        if len(words & set(_words(lowered[start:start + window]))) >= 0.7 * len(words):
            return start, "fuzzy"
    return None, "none"


def _hits(path: Path, kv: str) -> dict[str, bool] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "ok":
        return None
    for row in result["rows"]:
        if row["kv_precision"] == kv:
            return {q["id"]: bool(q["choice_hit"]) for q in row["questions"]}
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/size_sweep.yaml")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--quant-dir", default="int4_g64")
    parser.add_argument("--slots", type=int, default=1024)
    parser.add_argument("--tags", nargs="+", default=["s1024"])
    parser.add_argument("--kv", default="bf16")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bench = load_benchmark(cfg["benchmark_json"])
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    init_tag = f"s{args.slots}_init"
    runs = ["closed_book", "full_context", init_tag, *args.tags]

    label_of: dict[str, str] = {}
    found_by = defaultdict(int)
    hits: dict[str, dict[str, bool]] = defaultdict(dict)
    for p in cfg["patients"]:
        record = patient_record(bench, p)
        text = patient_text(record)
        prompt = context_system_prompt(text)
        enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        cutoff = enc["offset_mapping"][args.slots - 1][1] - prompt.index(text)
        print(f"lh_p{p:02d}: first {args.slots} tokens cover {cutoff} of {len(text)} record "
              f"chars ({cutoff / len(text):.1%})")
        for item in eval_items(record, p, cfg["questions_per_patient"]):
            pos, how = locate(item["correct_text"], text)
            found_by[how] += 1
            label_of[item["id"]] = "unlocated" if pos is None else (
                "prefix" if pos < cutoff else "later")
        for run in runs:
            kv = {"closed_book": "none"}.get(run, "bf16" if run == "full_context" else args.kv)
            h = _hits(run_dir(cfg, args.model, args.quant_dir, run, p) / "result.json", kv)
            if h is None:
                print(f"  missing {run} (KV {kv}) for lh_p{p:02d}")
                continue
            hits[run].update(h)

    counts = {lab: sum(v == lab for v in label_of.values()) for lab in _LABELS}
    print(f"\nanswers found: {dict(found_by)}; labels: {counts}\n")
    print(f"Correct per run (KV {args.kv} for cartridges):\n")
    print("| run | " + " | ".join(f"{lab} (n={counts[lab]})" for lab in _LABELS) + " | all |")
    print("|---|" + "---|" * (len(_LABELS) + 1))
    for run in runs:
        if not hits[run]:
            continue
        cells = [sum(hits[run].get(q, False) for q, lab in label_of.items() if lab == want)
                 for want in _LABELS]
        print(f"| {run} | " + " | ".join(map(str, cells)) + f" | {sum(cells)} |")

    print(f"\nTrained vs {init_tag}, trained-only / untrained-only:\n")
    print("| run | " + " | ".join(_LABELS) + " |")
    print("|---|" + "---|" * len(_LABELS))
    ref = hits[init_tag]
    for run in args.tags:
        if not hits[run] or not ref:
            continue
        cells = []
        for want in _LABELS:
            qs = [q for q, lab in label_of.items() if lab == want and q in ref and q in hits[run]]
            win = sum(hits[run][q] and not ref[q] for q in qs)
            loss = sum(ref[q] and not hits[run][q] for q in qs)
            cells.append(f"{win} / {loss}")
        print(f"| {run} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
