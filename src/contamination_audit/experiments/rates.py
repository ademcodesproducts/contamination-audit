"""Flag rates and their uncertainty.

Every flag rate in the results is a `Rate`, so there is exactly one place that
turns a count into a percentage. Intervals come from a nonparametric
bootstrap over benchmark items with a fixed seed, so reruns agree exactly.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

SEED = 42
RESAMPLES = 10_000


@dataclass(frozen=True)
class Rate:
    count: int
    total: int

    @property
    def fraction(self) -> float:
        return self.count / self.total

    @property
    def percent(self) -> float:
        return 100 * self.count / self.total

    def __str__(self) -> str:
        return f"{self.count} ({self.percent:.1f}%)"


def false_discovery_share(control: Rate, measurement: Rate) -> float | None:
    """The control rate over the measurement rate, or None if nothing was flagged.

    Control items cannot be in the corpus, so the control rate estimates how
    often the filter fires on text that is not there. Divided by the
    measurement rate, it bounds the share of measurement flags that are noise,
    provided both groups attract spurious matches equally often.
    """
    if not measurement.count:
        return None
    return control.fraction / measurement.fraction


def format_share(share: float | None) -> str:
    return "-" if share is None else f"{share:.2f}"


def central_interval(draws: Sequence[float]) -> tuple[float, float]:
    """The central 95% of a bootstrap distribution."""
    ordered = sorted(draws)
    n = len(ordered)
    return ordered[int(0.025 * n)], ordered[min(int(0.975 * n), n - 1)]


def bootstrap_rate(flagged: set[int], total: int) -> dict[str, float | list[float]]:
    """A flag rate with a 95% interval, resampling the observed items.

    Resamples the per-item flag indicator rather than simulating a binomial, so
    it assumes nothing about the shape of the distribution.
    """
    indicator = [1 if i in flagged else 0 for i in range(total)]
    rng = random.Random(SEED)
    draws = [100 * sum(rng.choices(indicator, k=total)) / total for _ in range(RESAMPLES)]
    low, high = central_interval(draws)
    return {"pct": Rate(len(flagged), total).percent, "ci": [low, high]}
