"""LongHealth corpus and eval items, read straight from benchmark_v5.json (read-only)."""

from __future__ import annotations

import json
import re
from pathlib import Path

_LETTERS = ("A", "B", "C", "D", "E")
_LEADING_LETTER_RE = re.compile(r"^[\s*]*\(?([A-E])\)?(?=[\s.:)*]|$)")
_OPTION_MARKER_RE = re.compile(r"(?:^|[\s*(:])([A-E])\)")
_ANSWER_LABEL_RE = re.compile(
    r"(?:answer|option)(?:\s+is)?[\s*:]*\(?([A-E])(?![a-z])", re.IGNORECASE
)
ANSWER_INSTRUCTION = "Answer with the letter and text of the single correct option."
# Bump when the eval prompt or scoring changes; older result.json files are re-run / skipped.
# 2: answer-format instruction in the prompt; accuracy = the chosen option.
EVAL_VERSION = 2


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
    """MCQ items with all five options inline, an answer-format instruction, and the key.

    ``correct_letter`` is None when the correct text does not match exactly one option; the
    choice is then judged by the chosen option's text instead.
    """
    items = []
    for idx, q in enumerate(record["questions"][:limit]):
        options = {k: q.get(f"answer_{k.lower()}", "").strip() for k in _LETTERS}
        options = {k: v for k, v in options.items() if v}
        correct = q.get("correct", "").strip()
        if not correct:
            raise ValueError(f"patient {patient} question {idx} has no correct answer.")
        matches = [k for k, v in options.items() if v == correct]
        opts = "\n".join(f"{k}) {v}" for k, v in options.items())
        items.append(
            {
                "id": f"lh_p{patient:02d}_q{idx:02d}",
                "prompt": f"{q['question']}\n{opts}\n\n{ANSWER_INSTRUCTION}",
                "options": {k: v.lower() for k, v in options.items()},
                "correct_text": correct.lower(),
                "correct_letter": matches[0] if len(matches) == 1 else None,
            }
        )
    return items


def chosen_option(answer: str, options: dict[str, str]) -> str | None:
    """The single option the answer commits to, or None.

    In order: a leading letter ("B", "**B)"), the first answer label ("**Answer: B**",
    "Correct Option: D"), the first "X)" marker ("The correct answer is: **C) ..."), the
    option whose full text appears earliest. Later mentions (explanations that walk through
    other options) are ignored.
    """
    match = (
        _LEADING_LETTER_RE.match(answer)
        or _ANSWER_LABEL_RE.search(answer)
        or _OPTION_MARKER_RE.search(answer)
    )
    if match and match.group(1) in options:
        return match.group(1)
    lowered = answer.lower()
    found = {}
    for k, v in options.items():
        m = re.search(rf"(?<!\w){re.escape(v)}(?!\w)", lowered)
        if m:
            found[k] = m.start()
    return min(found, key=found.get) if found else None


def score(answer: str, item: dict) -> dict:
    """Primary: the chosen option is the correct one. Logged: correct text appears anywhere."""
    choice = chosen_option(answer, item["options"])
    if item["correct_letter"] is not None:
        hit = choice == item["correct_letter"]
    else:
        hit = choice is not None and item["options"][choice] == item["correct_text"]
    return {
        "choice": choice,
        "choice_hit": hit,
        "substring_hit": item["correct_text"] in answer.lower(),
    }
