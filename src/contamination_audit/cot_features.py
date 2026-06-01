"""Chain-of-thought feature extraction (Ablation 3 / Table 7 / Figure 2).

STATUS: stub. The CoT analysis was produced in
``notebooks/cot_features.ipynb`` and not yet linearized.

Features computed in the paper (per trace, on original-perturbation only):
  - word_count: whitespace-delimited token count of full_trace
  - uncertainty_count: matches of {maybe, perhaps, possibly, unsure, not sure, uncertain}
  - hedging_count: matches of {might, could, should, would, may}
  - self_correction_count: matches of {actually, wait, let me reconsider,
                                       on second thought, scratch that, I made a mistake}
  - math_token_density: fraction of tokens containing a digit / operator
  - type_token_ratio: |unique tokens| / |tokens|

Mann-Whitney U is used for per-feature significance; report values are
exploratory (do not survive multiple-comparison correction).
"""

from __future__ import annotations


def compute_features(trace: str) -> dict:
    raise NotImplementedError("Pending notebook integration (Ablation 3).")
