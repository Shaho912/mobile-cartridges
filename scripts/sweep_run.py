#!/usr/bin/env python3
"""One size-sweep run for one (model, quantization, patient).

Modes:
  closed_book   question only, no context (guessing / parametric floor)
  full_context  whole patient record in the prompt (ceiling for this model + quantization)
  cartridge     teacher = the same fake-quantized model with the full record in context;
                cartridge trained against that same quantized base, then evaluated at each
                KV storage precision in --kv-eval

Before any weights load, every prompt is tokenized with this model's tokenizer and the run
fails loudly (exit 2, status "does_not_fit" in result.json) if anything exceeds the context
window. Nothing is ever truncated.

Usage:
    python scripts/sweep_run.py --config configs/size_sweep.yaml \\
        --model Qwen/Qwen3-0.6B --quant int4 --mode cartridge --slots 512 \\
        --kv-eval bf16 int8 --patient 1
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from cartridges.benchmarks.text_benchmark import (  # noqa: E402
    BOOTSTRAP_ANSWER_PROMPT,
    build_training_dataset,
    context_system_prompt,
)
from cartridges.core import TrainableKVCartridge  # noqa: E402
from cartridges.sweep.budget import MB  # noqa: E402
from cartridges.sweep.config import (  # noqa: E402
    bootstrap_path,
    load_config,
    run_dir,
    supervision_path,
)
from cartridges.sweep.kv_quant import kv_bytes_per_slot, with_kv_precision  # noqa: E402
from cartridges.sweep.longhealth import (  # noqa: E402
    EVAL_VERSION,
    eval_items,
    load_benchmark,
    patient_record,
    patient_text,
    score,
)
from cartridges.sweep.models import (  # noqa: E402
    PromptAdapter,
    attention_info,
    cache_visibility,
    clean_answer,
    greedy_generate,
    load_model,
)
from cartridges.train.cartridge import train_cartridge  # noqa: E402


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fail_fit(result_path: Path, base: dict, reason: str, needed: int, limit: int) -> int:
    payload = {**base, "status": "does_not_fit", "reason": reason,
               "tokens_needed": needed, "context_limit": limit}
    _write(result_path, payload)
    print(f"\nDOES NOT FIT: {reason}: needs {needed} tokens, context window is {limit}.",
          file=sys.stderr)
    return 2


def _eval(model, adapter, items, build_input, device, max_new_tokens):
    """build_input(item) -> (input_ids, cache). Returns per-question records."""
    records = []
    for item in items:
        input_ids, cache = build_input(item)
        t0 = time.perf_counter()
        ids = greedy_generate(model, input_ids.to(device), cache, adapter.stop_ids, max_new_tokens)
        answer = clean_answer(adapter.tokenizer.decode(ids, skip_special_tokens=True))
        hits = score(answer, item)
        records.append({"id": item["id"], "answer": answer, **hits,
                        "seconds": time.perf_counter() - t0})
        mark = "+" if hits["choice_hit"] else "-"
        print(f"  {mark} {item['id']}: chose {hits['choice']}  {answer[:100]!r}  "
              f"[expected {item['correct_letter']}) {item['correct_text'][:60]!r}]")
    return records


def _row(base: dict, kv: str, records: list[dict], weight_bytes: int, kv_bytes: float,
         budget_mb: float) -> dict:
    n = len(records)
    correct = sum(r["choice_hit"] for r in records)
    substring = sum(r["substring_hit"] for r in records)
    no_choice = sum(r["choice"] is None for r in records)
    total = weight_bytes + kv_bytes
    return {
        **base,
        "kv_precision": kv,
        "n_questions": n,
        "correct": correct,
        "correct_substring": substring,
        "no_choice": no_choice,
        "accuracy": correct / n,
        "substring_accuracy": substring / n,
        "weight_mb": weight_bytes / MB,
        "cartridge_mb": kv_bytes / MB,
        "total_mb": total / MB,
        "over_budget": total > budget_mb * MB,
        "questions": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/size_sweep.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--quant", choices=["bf16", "int8", "int4"], required=True)
    parser.add_argument("--mode", choices=["closed_book", "full_context", "cartridge"], required=True)
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--slot-tag", default=None,
                        help="Run label for cartridge runs (default s<slots>; e.g. max_bf16kv).")
    parser.add_argument("--kv-eval", nargs="+", default=["bf16"], choices=["bf16", "int8"])
    parser.add_argument("--patient", type=int, required=True)
    parser.add_argument("--quant-group-size", type=int, default=None,
                        help="Override quant_group_size (quantized runs go to <quant>_g<N>/).")
    parser.add_argument("--train-steps", type=int, default=None,
                        help="Override train.steps (1 = prefix-initialized cartridge, untrained).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.mode == "cartridge" and not args.slots:
        parser.error("--slots is required for --mode cartridge")

    cfg = load_config(args.config)
    if args.quant_group_size:
        cfg["quant_group_size"] = args.quant_group_size
    if args.train_steps:
        cfg["train"]["steps"] = args.train_steps
    # Group size is always in the directory name, so changing the default never mixes sizes.
    quant_dir = args.quant if args.quant == "bf16" else f"{args.quant}_g{cfg['quant_group_size']}"
    tag = args.mode if args.mode != "cartridge" else (args.slot_tag or f"s{args.slots}")
    out = run_dir(cfg, args.model, quant_dir, tag, args.patient)
    result_path = out / "result.json"
    if result_path.exists() and not args.force:
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        status = previous.get("status")
        same_train = args.mode != "cartridge" or status != "ok" \
            or previous["rows"][0].get("train_config") == cfg["train"]
        if status in ("ok", "does_not_fit") and previous.get("eval_version") == EVAL_VERSION \
                and same_train:
            print(f"Already done ({status}): {result_path}")
            return 0
        print(f"Re-running stale result (status={status}, eval_version="
              f"{previous.get('eval_version')}, current {EVAL_VERSION}, same train config "
              f"{same_train}): {result_path}")
    out.mkdir(parents=True, exist_ok=True)

    slice_id = f"lh_p{args.patient:02d}"
    record = patient_record(load_benchmark(cfg["benchmark_json"]), args.patient)
    text = patient_text(record)
    items = eval_items(record, args.patient, cfg["questions_per_patient"])
    system_prompt = context_system_prompt(text)
    info = attention_info(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    adapter = PromptAdapter(tokenizer, args.model)
    max_new = cfg["eval"]["max_new_tokens"]
    limit = info["max_position_embeddings"]
    budget_mb = cfg["budget_mb"]

    base = {
        "eval_version": EVAL_VERSION,
        "model": args.model, "quant": args.quant, "group_size": cfg["quant_group_size"],
        "mode": args.mode, "tag": tag, "slots": args.slots, "patient": args.patient,
        "teacher": cfg["teacher"], "budget_mb": budget_mb, "attention": info,
        "slurm_job": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
        "slurm_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "python": platform.python_version()},
    }

    # ── Fit checks: tokenize everything with this model's tokenizer, never truncate ──────
    sys_msg = {"role": "system", "content": system_prompt}
    query_lens = [adapter.query_ids(None, it["prompt"]).shape[-1] for it in items]
    full_prompts = None
    if args.mode in ("full_context", "cartridge"):
        full_prompts = [adapter.chat_ids([sys_msg, {"role": "user", "content": it["prompt"]}])
                        for it in items]
        need = max(len(p) for p in full_prompts) + max_new
        if need > limit:
            return _fail_fit(result_path, base, "full record + longest eval question", need, limit)

    answer_records = None
    if args.mode == "cartridge":
        boot_file = bootstrap_path(cfg, args.patient)
        if not boot_file.exists():
            raise SystemExit(f"Missing {boot_file}. Run scripts/sweep_bootstrap.py first.")
        boot = json.loads(boot_file.read_text(encoding="utf-8"))
        answer_records = [
            {"question": ex["question"].strip(),
             "user_message": f"{ex['question'].strip()}\n\n{BOOTSTRAP_ANSWER_PROMPT}",
             "assistant_text": ex["expected_answer"].strip(),
             "expected_answer": ex["expected_answer"].strip()}
            for ex in boot["examples"]
        ]
        longest_user = max(answer_records, key=lambda r: len(r["user_message"]))["user_message"]
        longest_target = max(
            len(tokenizer.encode(r["assistant_text"], add_special_tokens=False)) + 1
            for r in answer_records
        )
        need = len(adapter.chat_ids([sys_msg, {"role": "user", "content": longest_user}])) \
            + longest_target
        if need > limit:
            return _fail_fit(result_path, base, "teacher prompt (full record + bootstrap question)",
                             need, limit)
        doc_tokens = adapter.num_text_tokens(adapter.init_text(system_prompt))
        base["doc_tokens"] = doc_tokens
        if args.slots > doc_tokens:
            return _fail_fit(result_path, base,
                             f"{args.slots} slots exceed the {doc_tokens}-token record used for "
                             "prefix initialization", args.slots, doc_tokens)
        need = args.slots + max(query_lens) + max_new
        if need > limit:
            return _fail_fit(result_path, base, "cartridge slots + longest query", need, limit)
    else:
        need = max(query_lens) + max_new
        if need > limit:
            return _fail_fit(result_path, base, "longest eval question", need, limit)

    # ── Load the (fake-quantized) model used as teacher, student base, and evaluator ─────
    t_load = time.perf_counter()
    model, tokenizer, quant_report = load_model(
        args.model, args.quant, cfg["quant_group_size"], args.device
    )
    base["quant_report"] = quant_report
    base["load_seconds"] = time.perf_counter() - t_load
    weight_bytes = quant_report["weight_bytes"]
    print(f"[{args.model} {args.quant}] analytic weights {weight_bytes / MB:.1f} MB, "
          f"rel. weight error {quant_report['weighted_rel_error']:.4f}")
    shape = (info["num_hidden_layers"], info["num_key_value_heads"], info["head_dim"])

    rows = []
    if args.mode == "closed_book":
        records = _eval(model, adapter, items,
                        lambda it: (adapter.query_ids(None, it["prompt"]), None),
                        args.device, max_new)
        rows.append(_row(base, "none", records, weight_bytes, 0, budget_mb))

    elif args.mode == "full_context":
        prompts = {it["id"]: p for it, p in zip(items, full_prompts, strict=True)}
        records = _eval(model, adapter, items,
                        lambda it: (torch.tensor([prompts[it["id"]]]), None),
                        args.device, max_new)
        mean_tokens = sum(len(p) for p in full_prompts) / len(full_prompts)
        kv = mean_tokens * kv_bytes_per_slot(*shape, "bf16")
        base["full_context_tokens_mean"] = mean_tokens
        rows.append(_row(base, "bf16", records, weight_bytes, kv, budget_mb))

    else:
        sup = supervision_path(cfg, args.model, args.quant, args.patient)
        if not sup.exists():
            sup.parent.mkdir(parents=True, exist_ok=True)
            tmp = sup.with_name(f"{sup.name}.tmp{os.getpid()}")
            t0 = time.perf_counter()
            build_training_dataset(
                corpus_text=text, slice_id=slice_id, answer_records=answer_records,
                output_path=tmp, device=args.device, top_logprobs=cfg["train"]["top_logprobs"],
                model=model, tokenizer=tokenizer, eos_token_id=adapter.turn_end_id,
                chat_prompt_ids=adapter.chat_ids,
            )
            os.replace(tmp, sup)
            base["supervision_seconds"] = time.perf_counter() - t0
        base["supervision_path"] = str(sup)

        tr = cfg["train"]
        summary = train_cartridge(
            dataset_path=sup, output_dir=out / "train", slice_id=slice_id, device=args.device,
            cartridge_tokens=args.slots, num_frozen_tokens=tr["num_frozen_tokens"],
            learning_rate=tr["learning_rate"], steps=tr["steps"], seed=tr["seed"],
            validation_interval=tr["validation_interval"],
            holdout_validation=tr.get("holdout_validation", False),
            initialization_text=adapter.init_text(system_prompt),
            model=model, tokenizer=tokenizer, prompt_fn=adapter.query_ids,
        )
        if not tr["keep_checkpoints"]:
            for key in ("final_cartridge_path", "checkpoint_path"):
                Path(summary[key]).unlink(missing_ok=True)
        base["train"] = {k: summary[k] for k in (
            "best_loss", "best_step", "initial_loss", "final_loss", "train_loop_seconds",
            "validation_history", "cartridge_path")}
        base["train_config"] = tr
        t = base["train"]
        val = t["validation_history"]
        print(f"[train] val loss {val[0]['validation_loss']:.4f} (step {val[0]['step']}) -> best "
              f"{t['best_loss']:.4f} (step {t['best_step']}/{tr['steps']}), "
              f"{t['train_loop_seconds']:.0f} s")

        cart = TrainableKVCartridge.load(summary["cartridge_path"], device=args.device)
        base["cache_visibility"] = cache_visibility(model, cart)
        vis = base["cache_visibility"]
        if vis["layers_seeing_all_slots"] < info["num_hidden_layers"]:
            print(f"WARNING: only {vis['layers_seeing_all_slots']}/{info['num_hidden_layers']} "
                  f"layers keep all {cart.num_tokens} slots (min kept: {vis['min_stored_slots']}).")

        for kv in args.kv_eval:
            stored = with_kv_precision(cart, kv)
            print(f"\n── eval: {args.slots} slots, KV {kv} ──")
            records = _eval(model, adapter, items,
                            lambda it, c=stored: (adapter.query_ids(None, it["prompt"]),
                                                  c.as_cache(model.config)),
                            args.device, max_new)
            kv_bytes = args.slots * kv_bytes_per_slot(*shape, kv)
            rows.append(_row(base, kv, records, weight_bytes, kv_bytes, budget_mb))

    _write(result_path, {"status": "ok", "eval_version": EVAL_VERSION, "rows": rows})
    for r in rows:
        print(f"\n[{tag} KV={r['kv_precision']}] {r['correct']}/{r['n_questions']} "
              f"(no option chosen: {r['no_choice']}, substring {r['correct_substring']})  "
              f"total {r['total_mb']:.1f} MB{'  OVER BUDGET' if r['over_budget'] else ''}")
    print(f"Saved {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
