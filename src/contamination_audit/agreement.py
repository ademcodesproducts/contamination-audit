"""Inter-annotator agreement.

Cohen's kappa is unstable when one category dominates: with skewed prevalence it
can report near-zero for annotators who agree on almost every item. Gwet's AC1 is
reported alongside it for that reason, and the two are expected to diverge on this
data. Neither is meaningful without the raw agreement rate, so all three are returned.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


CONTAMINATED = {"VERBATIM", "PARAPHRASE", "ANSWER_LEAK"}
NOT_CONTAMINATED = {"TEMPLATE", "SAME_SKILL", "UNRELATED"}


@dataclass(frozen=True)
class AgreementResult:
    n: int
    categories: int
    raw_agreement: float
    cohens_kappa: float
    gwets_ac1: float
    kappa_ci: tuple[float, float]
    ac1_ci: tuple[float, float]

    def __str__(self) -> str:
        return (
            f"n={self.n} categories={self.categories}\n"
            f"  raw agreement  {self.raw_agreement:.3f}\n"
            f"  Cohen's kappa  {self.cohens_kappa:.3f}  "
            f"[{self.kappa_ci[0]:.3f}, {self.kappa_ci[1]:.3f}]\n"
            f"  Gwet's AC1     {self.gwets_ac1:.3f}  "
            f"[{self.ac1_ci[0]:.3f}, {self.ac1_ci[1]:.3f}]"
        )


def _observed(a: list[str], b: list[str]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def _cohens_kappa(a: list[str], b: list[str], cats: list[str]) -> float:
    n = len(a)
    po = _observed(a, b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def _gwets_ac1(a: list[str], b: list[str], cats: list[str]) -> float:
    n, k = len(a), len(cats)
    if k < 2:
        return 0.0
    po = _observed(a, b)
    pe = sum(
        ((a.count(c) / n + b.count(c) / n) / 2) * (1 - (a.count(c) / n + b.count(c) / n) / 2)
        for c in cats
    ) / (k - 1)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def _bootstrap(a: list[str], b: list[str], cats: list[str], stat, iters: int, seed: int):
    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        sa, sb = [a[i] for i in idx], [b[i] for i in idx]
        present = sorted({*sa, *sb})
        vals.append(stat(sa, sb, present))
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(int(0.975 * len(vals)), len(vals) - 1)]
    return (lo, hi)


def agreement(
    labels_a: list[str],
    labels_b: list[str],
    iters: int = 10_000,
    seed: int = 42,
) -> AgreementResult:
    """Two annotators, paired labels, already aligned by pair id."""
    if len(labels_a) != len(labels_b):
        raise ValueError(f"unpaired: {len(labels_a)} vs {len(labels_b)}")
    if not labels_a:
        raise ValueError("no labels")

    cats = sorted({*labels_a, *labels_b})
    return AgreementResult(
        n=len(labels_a),
        categories=len(cats),
        raw_agreement=_observed(labels_a, labels_b),
        cohens_kappa=_cohens_kappa(labels_a, labels_b, cats),
        gwets_ac1=_gwets_ac1(labels_a, labels_b, cats),
        kappa_ci=_bootstrap(labels_a, labels_b, cats, _cohens_kappa, iters, seed),
        ac1_ci=_bootstrap(labels_a, labels_b, cats, _gwets_ac1, iters, seed + 1),
    )


def to_binary(labels: list[str]) -> list[str]:
    """Collapse the label set to contaminated / not, for the headline number."""
    out = []
    for lab in labels:
        if lab in CONTAMINATED:
            out.append("CONTAMINATED")
        elif lab in NOT_CONTAMINATED:
            out.append("CLEAN")
        else:
            out.append(lab)
    return out
