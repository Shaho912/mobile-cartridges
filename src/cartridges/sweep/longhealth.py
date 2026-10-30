"""LongHealth corpus and eval items, read straight from benchmark_v5.json (read-only)."""

from __future__ import annotations

import json
import re
from pathlib import Path

_LETTERS = ("A", "B", "C", "D", "E")
_LEADING_LETTER_RE = re.compile(r"^\s*\(?([A-E])\)?(?=[\s.:)]|$)")
_OPTION_MARKER_RE = re.compile(r"(?:^|[\s*(:])([A-E])\)")


def load_benchmark(path: str | Path) -> dict:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def patient_record(benchmark: dict, patient: int) -> dict:
    key = f"patient_{patient:02d}"
    if key not in benchmark:
        raise KeyError(f"{key} not in LongHealth benchmark.")
    return benchmark[key]


def patient_text(record: dict) -> str:
    """Same joining as scripts/prepare_longhealth.py; never truncated."""
    texts = record["texts"]
    parts = [texts[k] for k in sorted(texts)]
    return "\n\n---\n\n".join(p.strip() for p in parts if p.strip())


def eval_items(record: dict, patient: int, limit: int | None = None) -> list[dict]:
    """MCQ items with all five options inline and the correct option's letter and text.

    ``correct_letter`` is None when the correct text does not match exactly one option; the
    letter metric then counts that item as a miss and the item is flagged.
    """
    items = []
    for idx, q in enumerate(record["questions"][:limit]):
        options = {k: q.get(f"answer_{k.lower()}", "") for k in _LETTERS}
        correct = q.get("correct", "").strip()
        if not correct:
            raise ValueError(f"patient {patient} question {idx} has no correct answer.")
        matches = [k for k, v in options.items() if v.strip() == correct]
        opts = "\n".join(f"{k}) {v}" for k, v in options.items() if v)
        items.append(
            {
                "id": f"lh_p{patient:02d}_q{idx:02d}",
                "prompt": f"{q['question']}\n{opts}",
                "correct_text": correct.lower(),
                "correct_letter": matches[0] if len(matches) == 1 else None,
            }
        )
    return items


def score(answer: str, item: dict) -> dict:
    """Primary: correct option text appears in the answer.

    Secondary: the chosen letter matches, where the choice is a leading letter ("B", "(B)")
    or else the first "X)" option marker ("The correct answer is: **C) ...").
    """
    match = _LEADING_LETTER_RE.match(answer) or _OPTION_MARKER_RE.search(answer)
    return {
        "substring_hit": item["correct_text"] in answer.lower(),
        "letter_hit": bool(match) and item["correct_letter"] is not None
        and match.group(1) == item["correct_letter"],
    }
