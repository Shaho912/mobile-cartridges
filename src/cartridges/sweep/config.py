"""Sweep config loading and run-directory layout."""

from __future__ import annotations

from pathlib import Path

import yaml

from cartridges.sweep.models import model_slug


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["benchmark_json"] = str(Path(cfg["benchmark_json"]).expanduser())
    cfg["output_root"] = Path(cfg["output_root"]).expanduser()
    return cfg


def run_dir(cfg: dict, model_id: str, quant: str, tag: str, patient: int) -> Path:
    return cfg["output_root"] / "runs" / model_slug(model_id) / quant / tag / f"lh_p{patient:02d}"


def bootstrap_path(cfg: dict, patient: int) -> Path:
    return cfg["output_root"] / "bootstrap" / f"lh_p{patient:02d}.json"


def supervision_path(cfg: dict, model_id: str, quant: str, patient: int) -> Path:
    group = f"_g{cfg['quant_group_size']}" if quant != "bf16" else ""
    name = f"{quant}{group}_teacher-{cfg['teacher']}"
    return cfg["output_root"] / "supervision" / model_slug(model_id) / name / f"lh_p{patient:02d}.jsonl"
