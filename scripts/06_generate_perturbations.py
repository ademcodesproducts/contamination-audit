"""Stage 6 (STUB): generate number_swap + surface_noise perturbations.

Linearizes notebooks/generate_perturbations.ipynb. See
src/contamination_audit/perturbation.py for the contract.

Inputs:  data/processed/{tulu,openthoughts}_c_sem.jsonl, data/processed/clean_baseline.jsonl
Outputs: data/processed/contaminated_and_perturbations.csv
"""

from __future__ import annotations

import sys

import _common  # noqa: F401


def main() -> None:
    raise NotImplementedError(
        "Pending notebook integration — see src/contamination_audit/perturbation.py "
        "and notebooks/README.md for the expected output schema."
    )


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print(f"[stub] {exc}", file=sys.stderr)
        sys.exit(2)
