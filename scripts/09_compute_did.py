"""Stage 9 (STUB): compute DiD estimates + cluster bootstrap (Tables 4 + 5).

Linearizes notebooks/compute_did.ipynb. Build on
src/contamination_audit/stats.cluster_bootstrap and stats.welch_t plus the
estimator skeleton in src/contamination_audit/did.py.

Inputs:  results/traces/{model}_traces_judged.jsonl
Outputs: results/tables/table4_accuracy.csv, results/tables/table5_did.csv
"""

from __future__ import annotations

import sys

import _common  # noqa: F401


def main() -> None:
    raise NotImplementedError(
        "Pending notebook integration — see src/contamination_audit/did.py "
        "and notebooks/README.md."
    )


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print(f"[stub] {exc}", file=sys.stderr)
        sys.exit(2)
