#!/usr/bin/env python3
"""Aggregate size-sweep results: table (CSV + markdown) and accuracy vs total MB plot.

Accuracy is computed per patient, then averaged; SE = sample std / sqrt(n_patients). Raw
correct/total counts are summed over patients so the noise level stays visible.

Usage:
    python scripts/sweep_aggregate.py --config configs/size_sweep.yaml
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from cartridges.sweep.config import load_config, run_dir  # noqa: E402
from cartridges.sweep.longhealth import (  # noqa: E402
    EVAL_VERSION,
    eval_items,
    load_benchmark,
    patient_record,
    score,
)

QUANT_ORDER = {"bf16": 0, "int8": 1, "int4": 2}
MODE_ORDER = {"closed_book": 0, "full_context": 1, "cartridge": 2}
MODEL_COLORS = ["#6b7280", "#2a78d6", "#eb6834", "#2f9e6e"]   # repo plot palette
QUANT_MARKERS = {"bf16": "o", "int8": "s", "int4": "^"}
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"


def _collect(cfg: dict) -> tuple[list[dict], list[dict]]:
    ok_rows, not_ok = [], []
    for path in sorted((cfg["output_root"] / "runs").glob("*/*/*/lh_p*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("eval_version") != EVAL_VERSION:
            row = (payload.get("rows") or [payload])[0]
            not_ok.append({**row, "status": "stale",
                           "reason": f"eval_version {payload.get('eval_version')}, rerun",
                           "path": str(path)})
        elif payload.get("status") == "ok":
            ok_rows.extend(payload["rows"])
        else:
            not_ok.append({**payload, "path": str(path)})
    return ok_rows, not_ok


def _rescore(rows: list[dict], cfg: dict) -> None:
    """Re-score every saved answer with the current parser, so parser fixes need no rerun."""
    bench = load_benchmark(cfg["benchmark_json"])
    items = {}
    for p in {r["patient"] for r in rows}:
        for it in eval_items(patient_record(bench, p), p, cfg["questions_per_patient"]):
            items[it["id"]] = it
    for r in rows:
        qs = r["questions"]
        for q in qs:
            q.update(score(q["answer"], items[q["id"]]))
        n = len(qs)
        r["correct"] = sum(q["choice_hit"] for q in qs)
        r["correct_substring"] = sum(q["substring_hit"] for q in qs)
        r["no_choice"] = sum(q["choice"] is None for q in qs)
        r["accuracy"] = r["correct"] / n
        r["substring_accuracy"] = r["correct_substring"] / n


def _group(rows: list[dict], model_order: dict) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["quant"], r["mode"], r["tag"], r["kv_precision"])].append(r)
    out = []
    for (model, quant, mode, tag, kv), rs in groups.items():
        accs = [r["accuracy"] for r in rs]
        n = len(rs)
        se = statistics.stdev(accs) / math.sqrt(n) if n > 1 else float("nan")
        slots = sorted({r["slots"] for r in rs if r["slots"] is not None})
        out.append({
            "model": model, "quant": quant, "mode": mode, "tag": tag, "kv_precision": kv,
            "slots": (str(slots[0]) if len(slots) == 1 else f"{min(slots)}-{max(slots)}")
            if slots else "",
            "slots_mean": statistics.mean(r["slots"] for r in rs) if slots else 0,
            "weight_mb": statistics.mean(r["weight_mb"] for r in rs),
            "cartridge_mb": statistics.mean(r["cartridge_mb"] for r in rs),
            "total_mb": statistics.mean(r["total_mb"] for r in rs),
            "over_budget": any(r["over_budget"] for r in rs),
            "acc_mean": statistics.mean(accs),
            "acc_se": se,
            "correct": sum(r["correct"] for r in rs),
            "questions": sum(r["n_questions"] for r in rs),
            "substring_acc_mean": statistics.mean(r["substring_accuracy"] for r in rs),
            "no_choice": sum(r["no_choice"] for r in rs),
            "n_patients": n,
            "patients": " ".join(str(r["patient"]) for r in sorted(rs, key=lambda r: r["patient"])),
        })
    out.sort(key=lambda g: (model_order.get(g["model"], 99), QUANT_ORDER[g["quant"]],
                            MODE_ORDER[g["mode"]], g["slots_mean"], g["kv_precision"]))
    return out


def _fmt_table(groups: list[dict]) -> list[str]:
    lines = [
        "| model | weights | mode | slots | KV | weight MB | cartridge MB | total MB | "
        "accuracy (mean ± SE) | raw | no choice | substring acc | patients |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in groups:
        se = "n/a" if math.isnan(g["acc_se"]) else f"{g['acc_se']:.3f}"
        flag = " OVER" if g["over_budget"] else ""
        lines.append(
            f"| {g['model']} | {g['quant']} | {g['mode'] if g['mode'] != 'cartridge' else g['tag']} "
            f"| {g['slots']} | {g['kv_precision']} | {g['weight_mb']:.1f} | {g['cartridge_mb']:.1f} "
            f"| {g['total_mb']:.1f}{flag} | {g['acc_mean']:.3f} ± {se} "
            f"| {g['correct']}/{g['questions']} | {g['no_choice']} | {g['substring_acc_mean']:.3f} "
            f"| {g['n_patients']} |"
        )
    return lines


def _plot(groups: list[dict], model_ids: list[str], budget_mb: float, path: Path) -> None:
    colors = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(model_ids)}
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    series: dict[tuple, list[dict]] = defaultdict(list)
    for g in groups:
        if g["mode"] == "cartridge":
            series[(g["model"], g["quant"], g["kv_precision"])].append(g)
    for (model, quant, kv), gs in series.items():
        gs = sorted(gs, key=lambda g: g["total_mb"])
        int8kv = kv == "int8"
        ax.errorbar(
            [g["total_mb"] for g in gs], [g["acc_mean"] for g in gs],
            yerr=[0 if math.isnan(g["acc_se"]) else g["acc_se"] for g in gs],
            color=colors[model], marker=QUANT_MARKERS[quant], markersize=8,
            linewidth=1.6, linestyle="--" if int8kv else "-", capsize=2.5, elinewidth=1,
            markerfacecolor="white" if int8kv else colors[model], markeredgewidth=1.4, zorder=3,
        )
    for g in groups:
        if g["mode"] == "cartridge":
            continue
        ax.errorbar(
            [g["total_mb"]], [g["acc_mean"]],
            yerr=[[0 if math.isnan(g["acc_se"]) else g["acc_se"]]] * 2,
            color=colors[g["model"]], marker="*" if g["mode"] == "full_context" else "X",
            markersize=13 if g["mode"] == "full_context" else 9, linestyle="none",
            capsize=2.5, elinewidth=1, markeredgecolor="white", markeredgewidth=0.8, zorder=4,
        )

    ax.set_xscale("log")
    xmin, xmax = ax.get_xlim()
    ax.axvspan(budget_mb, xmax, color=GRID, alpha=0.45, zorder=0, linewidth=0)
    ax.axvline(budget_mb, color=INK, linewidth=1.4, zorder=2)
    ax.text(budget_mb * 0.97, 1.02, f"{budget_mb:.0f} MB budget", ha="right", va="bottom",
            fontsize=9, color=INK)
    ax.axhline(0.2, color=MUTED, linewidth=1, linestyle=":", zorder=1)
    ax.text(xmax / 1.05, 0.19, "chance (1 of 5 options)", fontsize=8, color=MUTED, va="top",
            ha="right")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_xlabel("Total memory: weights + cartridge KV (MB, log scale)", fontsize=10, color=MUTED)
    ax.set_ylabel("LongHealth accuracy (mean ± SE over patients)", fontsize=10, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    model_handles = [mlines.Line2D([], [], color=colors[m], linewidth=2, label=m) for m in model_ids
                     if any(g["model"] == m for g in groups)]
    quant_handles = [mlines.Line2D([], [], color=INK, marker=mk, linestyle="none", markersize=7,
                                   label=f"weights {q}") for q, mk in QUANT_MARKERS.items()]
    mode_handles = [
        mlines.Line2D([], [], color=INK, linewidth=1.6, label="cartridge, slot sweep (bf16 KV)"),
        mlines.Line2D([], [], color=INK, linewidth=1.6, linestyle="--", marker="o",
                      markerfacecolor="white", label="cartridge, int8 KV"),
        mlines.Line2D([], [], color=INK, marker="*", linestyle="none", markersize=11,
                      label="full context (KV = whole record)"),
        mlines.Line2D([], [], color=INK, marker="X", linestyle="none", markersize=8,
                      label="closed book (no context)"),
    ]
    ax.legend(handles=model_handles + quant_handles + mode_handles, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_title("Accuracy vs memory per chip, single cartridge per patient",
                 fontsize=11, fontweight="bold", loc="left", color=INK)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/size_sweep.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_ids = [m["id"] for m in cfg["models"]]
    rows, not_ok = _collect(cfg)
    if not rows:
        print("No completed runs found.", file=sys.stderr)
        return 1
    _rescore(rows, cfg)
    groups = _group(rows, {m: i for i, m in enumerate(model_ids)})

    report = cfg["output_root"] / "report"
    report.mkdir(parents=True, exist_ok=True)
    with (report / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)

    lines = [f"# Size sweep results (budget {cfg['budget_mb']} MB, weights + cartridge KV)", "",
             "Accuracy = the option the answer commits to (leading letter, else first 'X)' marker, "
             "else earliest option text) is correct. No choice = answers that picked no option. "
             "Substring acc (logged only) = correct option text appears anywhere, inflated by "
             "explanations. OVER = total exceeds the budget. Totals exclude activations.",
             "", *_fmt_table(groups)]

    if not_ok:
        lines += ["", "## Runs that did not complete", "",
                  "| model | weights | run | patient | status | reason |", "|---|---|---|---|---|---|"]
        for r in not_ok:
            lines.append(f"| {r.get('model')} | {r.get('quant')} | {r.get('tag')} | "
                         f"{r.get('patient')} | {r.get('status')} | {r.get('reason', '')} |")

    manifest = cfg["output_root"] / "jobs" / "manifest.json"
    if manifest.exists():
        missing = []
        for j in json.loads(manifest.read_text(encoding="utf-8")):
            tag = j.get("slot_tag") or j["mode"]
            if not (run_dir(cfg, j["model"], j["quant"], tag, j["patient"]) / "result.json").exists():
                missing.append(f"{j['model']} {j['quant']} {tag} p{j['patient']:02d}")
        if missing:
            lines += ["", f"## Missing runs ({len(missing)})", "", *[f"- {m}" for m in missing]]

    gguf = sorted((cfg["output_root"] / "gguf").glob("*_sizes.json"))
    if gguf:
        lines += ["", "## GGUF file sizes (llama.cpp cross-check)", "",
                  "| model | bf16 MB | Q8_0 MB | Q4_K_M MB |", "|---|---|---|---|"]
        for f in gguf:
            d = json.loads(f.read_text(encoding="utf-8"))
            b = d["bytes"]
            lines.append(f"| {d['model']} | {b['bf16'] / 1e6:.1f} | {b['Q8_0'] / 1e6:.1f} | "
                         f"{b['Q4_K_M'] / 1e6:.1f} |")

    (report / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot(groups, model_ids, cfg["budget_mb"], report / "accuracy_vs_mb.png")
    print("\n".join(lines))
    print(f"\nSaved {report / 'results.csv'}, results.md, accuracy_vs_mb.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
