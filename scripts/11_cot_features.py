"""Stage 11 (STUB): CoT feature extraction + Mann-Whitney U (Table 7 + Figure 2).

Computes word count, hedging, self-correction, math token density, type-token
ratio per trace; runs per-feature non-parametric significance tests.

Linearizes notebooks/cot_features.ipynb. Use src/contamination_audit/cot_features.py.

Inputs:  results/traces/openthoughts_traces_judged.jsonl
Outputs: results/tables/table7_cot_features.csv, results/figures/figure2_cot_distributions.png
"""

from __future__ import annotations

import sys

import _common  # noqa: F401


def main() -> None:
    raise NotImplementedError(
        "Pending notebook integration — see src/contamination_audit/cot_features.py."
    )


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print(f"[stub] {exc}", file=sys.stderr)
        sys.exit(2)
