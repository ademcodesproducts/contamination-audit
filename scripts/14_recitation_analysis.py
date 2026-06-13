"""Stage 14: recitation vs shortcut breakdown (paper §7.2 / Ablation 2).

For each (model, split) on the number_swap perturbation, classifies traces as
recited / correct / neither / null / ambiguous and reports the recitation rate.

Reads:  results/traces/{model}_traces_judged.jsonl
Writes: results/tables/table_recitation.csv
"""

from __future__ import annotations

import argparse
import logging

import _common  # noqa: F401

from contamination_audit.config import configure_logging
from contamination_audit.io import REPO_ROOT, load_jsonl
from contamination_audit.recitation import per_model_breakdown

_log = logging.getLogger("recitation")

DEFAULT_MODELS = ["openthoughts", "tulu", "s1"]


def _load(models: list[str]) -> list[dict]:
    base = REPO_ROOT / "results" / "traces"
    out: list[dict] = []
    for model in models:
        for name in (f"{model}_traces_judged.jsonl", f"{model}_traces.jsonl"):
            path = base / name
            if path.exists():
                records = load_jsonl(path)
                for r in records:
                    r.setdefault("model", model)
                out.extend(records)
                _log.info("loaded %d %s records from %s", len(records), model, path.name)
                break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    configure_logging()
    records = _load(args.models)
    if not records:
        _log.error("no trace data found — run scripts/07_run_inference.py first")
        return

    df = per_model_breakdown(records)
    out = REPO_ROOT / "results" / "tables" / "table_recitation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    _log.info("wrote %s", out)


if __name__ == "__main__":
    main()
