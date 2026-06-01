"""Stage 7 (STUB): run OpenThinker-7B / Tulu-3-8B-SFT on perturbed prompts.

Linearizes notebooks/run_inference.ipynb. See
src/contamination_audit/inference.py for the contract.

Inputs:  data/processed/contaminated_and_perturbations.csv
Outputs: results/traces/{model}_traces.jsonl  (already populated for openthoughts + tulu)
"""

from __future__ import annotations

import sys

import _common  # noqa: F401


def main() -> None:
    raise NotImplementedError(
        "Pending notebook integration — see src/contamination_audit/inference.py "
        "and notebooks/README.md."
    )


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print(f"[stub] {exc}", file=sys.stderr)
        sys.exit(2)
