"""Bootstrap and significance utilities for the paper's behavioral analysis.

Provides:
  - ``proportion_bootstrap``: 95% CI on a Bernoulli rate (used by Table 2/Fig 1).
  - ``cluster_bootstrap``: cluster-level resampling for the DiD estimator (Table 5).
  - ``welch_t``: Welch's two-sample t-test on per-problem accuracy differences.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def proportion_bootstrap(
    n_hits: int,
    n_total: int,
    *,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (low, high) percentile CI on the rate ``n_hits / n_total``."""
    rng = np.random.default_rng(seed)
    trials = np.zeros(n_total, dtype=np.int8)
    trials[:n_hits] = 1
    samples = rng.choice(trials, size=(n_bootstrap, n_total), replace=True).mean(axis=1)
    lo = float(np.percentile(samples, alpha / 2 * 100))
    hi = float(np.percentile(samples, (1 - alpha / 2) * 100))
    return lo, hi


def cluster_bootstrap(
    deltas: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Cluster-bootstrap a mean.

    ``deltas`` is a 1-D array of per-cluster (per-problem) values. Returns
    ``(low, high, samples)`` where ``samples`` is the bootstrap distribution
    of the mean, useful for downstream computation (e.g. DiD).
    """
    rng = np.random.default_rng(seed)
    n = len(deltas)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = deltas[idx].mean()
    lo = float(np.percentile(samples, alpha / 2 * 100))
    hi = float(np.percentile(samples, (1 - alpha / 2) * 100))
    return lo, hi, samples


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's two-sample t-test. Returns ``(t_stat, p_value)``."""
    result = stats.ttest_ind(a, b, equal_var=False)
    return float(result.statistic), float(result.pvalue)
