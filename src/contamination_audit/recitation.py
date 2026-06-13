"""Recitation analysis (paper §7.2 / Ablation 2).

For each contaminated and clean problem, on the **number_swap** perturbation:

  - ``null``      — model produced no extractable answer
  - ``ambiguous`` — original ground truth equals the perturbed ground truth
                    (perturbation was a no-op for this problem)
  - ``recited``   — model's answer equals the *original* ground truth, not the
                    perturbed one. Direct evidence of memorization.
  - ``correct``   — model's answer equals the perturbed ground truth (it
                    actually re-solved the modified problem).
  - ``neither``   — answer matches neither (genuine wrong answer).

Recitation rate is reported as ``recited / (total - null - ambiguous)`` so it
measures the model's behavior on cases where memorization is observable.

Paper §7.2 finding: neither model shows any recitation on contaminated
problems — contamination operates as a *shortcut pattern* (familiar solution
trajectory) rather than rote key-value recall.

Linearised from cell 16 of ``downloads/analysis.ipynb``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd


def _normalize(value) -> str | None:
    return str(value).strip().lower() if value is not None else None


def per_model_breakdown(records: Iterable[dict]) -> pd.DataFrame:
    """Return one row per (model, split) with the five recitation categories + rate."""
    records = list(records)

    # Build a lookup: (math500_id, split) → original ground truth (normalized).
    orig_gt: dict[tuple[str, str], str | None] = {
        (r["math500_id"], r["split"]): _normalize(r.get("ground_truth"))
        for r in records
        if r.get("perturbation_type") == "original"
    }

    buckets: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"recited": 0, "correct": 0, "neither": 0, "null": 0, "ambiguous": 0, "total": 0}
    )

    for r in records:
        if r.get("perturbation_type") != "number_swap":
            continue
        model = r.get("model") or r.get("dataset")
        split = r.get("split")
        key = (model, split)

        original = orig_gt.get((r["math500_id"], split))
        perturbed = _normalize(r.get("ground_truth"))
        final = _normalize(r.get("final_answer"))

        buckets[key]["total"] += 1
        if final is None:
            buckets[key]["null"] += 1
        elif original == perturbed:
            buckets[key]["ambiguous"] += 1
        elif final == original:
            buckets[key]["recited"] += 1
        elif final == perturbed:
            buckets[key]["correct"] += 1
        else:
            buckets[key]["neither"] += 1

    rows = []
    for (model, split), b in buckets.items():
        evaluable = b["total"] - b["ambiguous"] - b["null"]
        rate = b["recited"] / evaluable if evaluable else float("nan")
        rows.append({
            "model": model,
            "split": split,
            **b,
            "recitation_rate": round(rate, 4) if evaluable else None,
        })

    return pd.DataFrame(rows).sort_values(["model", "split"]).reset_index(drop=True)
