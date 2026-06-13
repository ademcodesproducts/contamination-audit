"""Null-rate analysis (paper §7.1 / Table 6 / Figure 1).

A "null" trace is one where the model failed to produce an extractable
``\\boxed{}`` answer at all — i.e. ``final_answer is None`` on the record. The
paper's clearest contamination signal is the 37.5% null rate on the
OpenThoughts contaminated × number_swap cell vs 20.8% on the clean baseline:
contamination shows up as model abandonment under perturbation, not as
incorrect answers.

Linearised from cells 6 + 19 + 20 of ``downloads/analysis.ipynb``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd


def per_cell_null_rate(records: Iterable[dict]) -> pd.DataFrame:
    """Return a long-format DataFrame: (model, split, perturbation_type, total, nulls, null_pct)."""
    counts: dict[tuple, dict[str, int]] = defaultdict(lambda: {"total": 0, "nulls": 0})
    for r in records:
        key = (r.get("model") or r.get("dataset"), r.get("split"), r.get("perturbation_type"))
        counts[key]["total"] += 1
        if r.get("final_answer") is None:
            counts[key]["nulls"] += 1

    rows = []
    for (model, split, ptype), c in counts.items():
        total = c["total"]
        nulls = c["nulls"]
        rows.append({
            "model": model,
            "split": split,
            "perturbation_type": ptype,
            "total": total,
            "nulls": nulls,
            "null_pct": round(nulls / total * 100, 2) if total else 0.0,
        })

    return pd.DataFrame(rows).sort_values(["model", "split", "perturbation_type"]).reset_index(drop=True)
