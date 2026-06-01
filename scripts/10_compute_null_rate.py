"""Stage 10 (STUB): null-rate analysis (Table 6 + Figure 1).

Computes the fraction of model outputs without an extractable ``\\boxed{}``
answer per (split, perturbation) cell. The 37.5% OpenThoughts contaminated /
number_swap cell is the paper's clearest contamination signal.

Linearizes notebooks/null_rate.ipynb.

Inputs:  results/traces/{model}_traces_judged.jsonl
Outputs: results/tables/table6_null_rate.csv, results/figures/figure1_null_rates.png
"""

from __future__ import annotations

import sys

import _common  # noqa: F401


def main() -> None:
    raise NotImplementedError(
        "Pending notebook integration — uses answer_extract.extract_boxed already in the library. "
        "See notebooks/README.md."
    )


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print(f"[stub] {exc}", file=sys.stderr)
        sys.exit(2)
