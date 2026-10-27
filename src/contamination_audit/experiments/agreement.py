"""How far the five filters agree, and how certain the headline rates are.

If the filters were finding the same contamination, their flags would overlap.
This counts how many filters flag each measurement item, bootstraps the share
flagged by exactly k filters, and gives every filter's flag rates a 95%
interval. The control set has only 280 items, so its rates carry real
uncertainty and should not be quoted bare.
"""

from __future__ import annotations

import random
from collections import Counter
from itertools import combinations

from ..errors import MissingDataError
from .base import Experiment
from .rates import RESAMPLES, SEED, Rate, bootstrap_rate, central_interval


class FilterAgreement(Experiment):
    name = "agreement"
    title = "Agreement between the five filters, with bootstrap intervals"

    def run(self):
        flag_counts = self.study.load_result("flag_counts")
        n_measure = flag_counts["n_measure"]

        flagged_by = {name: set(f["measure"]) for name, f in flag_counts["filters"].items()}
        flagged_by["decon"] = self._decon_flags()

        # How many filters flag each measurement item.
        votes = [sum(i in flagged for flagged in flagged_by.values()) for i in range(n_measure)]
        observed = Counter(k for k in votes if k)
        n_flagged = sum(observed.values())

        return {
            "flagged_any": n_flagged,
            "n": n_measure,
            "consensus": self._consensus(votes, observed, n_flagged, len(flagged_by)),
            "jaccard": self._pairwise_jaccard(flagged_by),
            "rates": {
                name: {
                    "measure": bootstrap_rate(set(f["measure"]), n_measure),
                    "control": bootstrap_rate(set(f["control"]), flag_counts["n_control"]),
                }
                for name, f in flag_counts["filters"].items()
            },
        }

    def _decon_flags(self) -> set[int]:
        """decon's measurement flags, which belong in the agreement denominator.

        The decon runner records counts, not item indices. An empty set is right
        exactly when that count is zero; otherwise agreement cannot be computed.
        """
        decon = self.study.load_result("decon")
        if decon["measure"]:
            raise MissingDataError(
                f"decon flagged {decon['measure']} measurement items, but its result "
                "records counts only; save the flagged indices before computing agreement"
            )
        return set()

    @staticmethod
    def _consensus(votes, observed, n_flagged, n_filters):
        """For each k, the share of flagged items exactly k filters flag, with a 95% interval."""
        n = len(votes)
        rng = random.Random(SEED)
        resampled = {k: [] for k in range(1, n_filters + 1)}
        for _ in range(RESAMPLES):
            resampled_votes = [votes[rng.randrange(n)] for _ in range(n)]
            sample = Counter(k for k in resampled_votes if k)
            total = sum(sample.values())
            for k in resampled:
                resampled[k].append(sample.get(k, 0) / total if total else 0.0)

        consensus = {}
        for k, draws in resampled.items():
            low, high = central_interval(draws)
            items = observed.get(k, 0)
            consensus[k] = {
                "items": items,
                "pct": Rate(items, n_flagged).percent,
                "ci": [100 * low, 100 * high],
            }
        return consensus

    @staticmethod
    def _pairwise_jaccard(flagged_by):
        firing = [name for name, flagged in flagged_by.items() if flagged]
        return {
            f"{a}|{b}": len(flagged_by[a] & flagged_by[b]) / len(flagged_by[a] | flagged_by[b])
            for a, b in combinations(firing, 2)
        }

    def report(self, result):
        lines = [
            f"{result['flagged_any']} of {result['n']} measurement items "
            "flagged by at least one filter"
        ]
        for k, row in result["consensus"].items():
            if row["items"]:
                low, high = row["ci"]
                lines.append(
                    f"  exactly {k}: {row['items']:4d} items  {row['pct']:5.1f}%  "
                    f"95% CI [{low:.1f}, {high:.1f}]"
                )
        lines.append(f"highest pairwise Jaccard: {max(result['jaccard'].values()):.3f}")
        lines.append(f"{'filter':28s} {'measurement':>22s} {'control':>22s}")
        for name, rates in result["rates"].items():
            cells = [
                f"{r['pct']:5.1f}% [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}]"
                for r in (rates["measure"], rates["control"])
            ]
            lines.append(f"{name:28s} {cells[0]:>22s} {cells[1]:>22s}")
        return lines
