"""Stage 9: compute DiD estimates (paper Tables 4 + 5).

Loads judged trace files for OpenThoughts, Tulu, and (if present) s1, computes:

  - Table 4 — per-cell accuracy: count + accuracy in every
              (model, split, perturbation_type) bucket
  - Table 5 — DiD point estimate + 10,000-iter cluster bootstrap 95% CI +
              Welch's t-test, per (model, perturbation_type) pair

Reads:  results/traces/{model}_traces_judged.jsonl  (fallback: {model}_traces.jsonl)
Writes: results/tables/table4_accuracy.csv
        results/tables/table5_did.csv
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

import _common  # noqa: F401

from contamination_audit.config import configure_logging, load_config
from contamination_audit.did import compute_did, inflation_from_did
from contamination_audit.io import REPO_ROOT, load_jsonl

_log = logging.getLogger("compute_did")


DEFAULT_MODELS = ["openthoughts", "tulu", "s1"]
PERTURBATIONS = ("number_swap", "surface_noise")


def _load_traces(model: str) -> list[dict]:
    """Prefer the judged file (has llm_correct); fall back to raw traces."""
    base = REPO_ROOT / "results" / "traces"
    for name in (f"{model}_traces_judged.jsonl", f"{model}_traces.jsonl"):
        path = base / name
        if path.exists():
            records = load_jsonl(path)
            for r in records:
                r.setdefault("model", model)
            _log.info("  %s: %d records from %s", model, len(records), path.name)
            return records
    _log.warning("no traces for %s under %s", model, base)
    return []


def build_table4(records_per_model: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for model, records in records_per_model.items():
        df = pd.DataFrame([
            {
                "split": r.get("split"),
                "perturbation_type": r.get("perturbation_type"),
                "level": pd.to_numeric(r.get("level"), errors="coerce"),
                "llm_correct": int(bool(r.get("llm_correct") or r.get("correct"))),
            }
            for r in records
            if r.get("split") and r.get("perturbation_type")
        ])
        if df.empty:
            continue
        summary = (
            df.groupby(["split", "perturbation_type"])
              .agg(n=("llm_correct", "count"),
                   correct=("llm_correct", "sum"),
                   accuracy=("llm_correct", "mean"),
                   avg_level=("level", "mean"))
              .reset_index()
        )
        summary["model"] = model
        rows.append(summary)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    cols = ["model", "split", "perturbation_type", "n", "correct", "accuracy", "avg_level"]
    return out[cols].sort_values(["model", "split", "perturbation_type"]).reset_index(drop=True)


def build_table5(
    records_per_model: dict[str, list[dict]],
    *,
    n_bootstrap: int,
    seed: int,
    benchmark_size: int = 500,
) -> pd.DataFrame:
    rows = []
    for model, records in records_per_model.items():
        n_contam_problems = len({r["math500_id"] for r in records if r.get("split") == "contaminated"})
        for perturb in PERTURBATIONS:
            result = compute_did(records, perturb, n_bootstrap=n_bootstrap, seed=seed)
            if result is None:
                _log.warning("  %s / %s: insufficient data for DiD", model, perturb)
                continue
            row = result.to_dict()
            row["model"] = model
            row["inflation_pp"] = round(
                inflation_from_did(result.did, n_contam_problems, benchmark_size) * 100, 3
            )
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    cols = [
        "model", "perturbation_type", "n_contaminated", "n_clean",
        "delta_C", "delta_N", "did", "ci_low", "ci_high", "t", "p", "inflation_pp",
    ]
    return pd.DataFrame(rows)[cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--config", default=str(_common.DEFAULT_CONFIG))
    args = parser.parse_args()

    configure_logging()
    app_config = load_config(args.config)

    records_per_model = {m: _load_traces(m) for m in args.models}
    records_per_model = {m: r for m, r in records_per_model.items() if r}

    if not records_per_model:
        _log.error("no trace data found — run scripts/07_run_inference.py first")
        return

    table4 = build_table4(records_per_model)
    table5 = build_table5(
        records_per_model,
        n_bootstrap=app_config.stats.n_bootstrap,
        seed=app_config.stats.seed,
    )

    out_dir = REPO_ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table4.to_csv(out_dir / "table4_accuracy.csv", index=False)
    table5.to_csv(out_dir / "table5_did.csv", index=False)

    _log.info("=== Table 4 — Accuracy by model × split × perturbation ===")
    print(table4.to_string(index=False))
    _log.info("=== Table 5 — Difference-in-differences estimates ===")
    print(table5.to_string(index=False))
    _log.info("wrote %s/table4_accuracy.csv and table5_did.csv", out_dir)


if __name__ == "__main__":
    main()
