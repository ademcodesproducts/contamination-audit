"""Difference-in-differences estimator (Section 5.4 of the paper).

STATUS: stub. The DiD computation, 10,000-iteration cluster bootstrap CIs, and
Welch's t-test that produce Tables 4 and 5 were run in
``notebooks/compute_did.ipynb`` and have not yet been linearized.

The estimator is:
    DiD = (a_C_orig - a_C_perturb) - (a_N_orig - a_N_perturb)
where a_C and a_N are mean per-problem accuracies in the contaminated and
clean splits, and ``perturb`` is either ``number_swap`` or ``surface_noise``.

When linearizing, build on ``stats.cluster_bootstrap`` and ``stats.welch_t``
already in place.

Expected outputs:
  results/tables/table4_accuracy.csv          (judge-graded accuracy per cell)
  results/tables/table5_did.csv               (delta_C, delta_N, DiD, 95% CI, t, p)
"""

from __future__ import annotations

import numpy as np

from .stats import cluster_bootstrap, welch_t  # noqa: F401  (re-exported for callers)


def estimate(
    contaminated_orig: np.ndarray,
    contaminated_perturb: np.ndarray,
    clean_orig: np.ndarray,
    clean_perturb: np.ndarray,
) -> dict:
    raise NotImplementedError("Pending notebook integration (Section 5.4).")
