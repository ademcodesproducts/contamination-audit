"""Difference-in-differences estimator (paper §5.4 / Tables 4 + 5).

Per-problem accuracy in the contaminated and clean splits is the binary
``llm_correct`` field on each trace record. We compute, per perturbation type:

    δ_C = mean_p [ acc(p, original) - acc(p, perturb) ]   over contaminated p
    δ_N = mean_p [ acc(p, original) - acc(p, perturb) ]   over clean p
    DiD = δ_C - δ_N

CIs come from a 10,000-iteration cluster bootstrap (resampling problems
independently within each split). Welch's t-test is reported alongside for
parametric comparison.

Linearised from cell 13 of ``downloads/analysis.ipynb`` — the only material
change is moving the patterns/constants into named functions for testability.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DidResult:
    perturbation_type: str
    n_contaminated: int          # number of contaminated problems contributing pairs
    n_clean: int                 # number of clean problems contributing pairs
    delta_contaminated: float    # mean(acc_original - acc_perturb) over contaminated
    delta_clean: float           # mean(acc_original - acc_perturb) over clean
    did: float                   # delta_contaminated - delta_clean
    ci_low: float
    ci_high: float
    t_stat: float
    p_value: float

    def to_dict(self) -> dict:
        return {
            "perturbation_type": self.perturbation_type,
            "n_contaminated": self.n_contaminated,
            "n_clean": self.n_clean,
            "delta_C": round(self.delta_contaminated, 4),
            "delta_N": round(self.delta_clean, 4),
            "did": round(self.did, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "t": round(self.t_stat, 4),
            "p": round(self.p_value, 4),
        }


def _pairwise_deltas(records: Iterable[dict], perturb_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(delta_contaminated, delta_clean)`` arrays of per-problem accuracy drops.

    A problem contributes a delta only if both its original and perturbed records
    have a ``llm_correct`` value (or fallback ``correct``).
    """
    by_problem: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for r in records:
        if r.get("perturbation_type") not in ("original", perturb_type):
            continue
        key = (r["math500_id"], r["split"])
        correct = r.get("llm_correct")
        if correct is None:
            correct = r.get("correct")
        if correct is None:
            continue
        by_problem[key][r["perturbation_type"]] = int(bool(correct))

    delta_c, delta_n = [], []
    for (_, split), ptypes in by_problem.items():
        if "original" not in ptypes or perturb_type not in ptypes:
            continue
        delta = ptypes["original"] - ptypes[perturb_type]
        (delta_c if split == "contaminated" else delta_n).append(delta)

    return np.asarray(delta_c, dtype=float), np.asarray(delta_n, dtype=float)


def compute_did(
    records: Iterable[dict],
    perturb_type: str,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> DidResult | None:
    """Compute the DiD point estimate plus a cluster-bootstrap CI and Welch's t."""
    delta_c, delta_n = _pairwise_deltas(records, perturb_type)
    if delta_c.size == 0 or delta_n.size == 0:
        return None

    did = float(delta_c.mean() - delta_n.mean())

    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boots[i] = (
            rng.choice(delta_c, delta_c.size, replace=True).mean()
            - rng.choice(delta_n, delta_n.size, replace=True).mean()
        )
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

    t_stat, p_value = stats.ttest_ind(delta_c, delta_n, equal_var=False)

    return DidResult(
        perturbation_type=perturb_type,
        n_contaminated=int(delta_c.size),
        n_clean=int(delta_n.size),
        delta_contaminated=float(delta_c.mean()),
        delta_clean=float(delta_n.mean()),
        did=did,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        t_stat=float(t_stat),
        p_value=float(p_value),
    )


def inflation_from_did(did: float, n_contaminated_total: int, n_total_benchmark: int = 500) -> float:
    """Project the DiD effect to a benchmark-level accuracy inflation estimate.

    Reported as a fraction of the benchmark size (multiply by 100 for percentage
    points). Paper §6 reports point estimates ≤ 0.4 pp.
    """
    return did * (n_contaminated_total / n_total_benchmark)
