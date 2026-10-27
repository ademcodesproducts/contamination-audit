"""The control set's two kinds of question, scored separately.

The measurement set is almost entirely competition problems, but only 69 of
the 280 controls are; the rest are questions derived from research papers.
Pooling the two into one false-positive rate averages populations the filters
treat very differently, weighted by which MathArena sets happen to exist.
"""

from __future__ import annotations

from collections import Counter

from ..data import DOLCI
from .base import Experiment
from .rates import Rate

COMPETITIONS_2026 = frozenset({"aime_2026", "hmmt_feb_2026", "usamo_2026"})
STRATA = ("competition", "research")


class ControlStrata(Experiment):
    name = "strata"
    title = "Control false positives, competition problems versus research-derived questions"

    def run(self):
        flag_counts = self.study.load_result("flag_counts")
        control = self.study.split(DOLCI).control

        stratum_of = [
            "competition" if item.benchmark in COMPETITIONS_2026 else "research" for item in control
        ]
        sizes = Counter(stratum_of)

        filters = {}
        for name, flagged in flag_counts["filters"].items():
            hits = Counter(stratum_of[i] for i in flagged["control"])
            filters[name] = {
                stratum: {
                    "n": sizes[stratum],
                    "flagged": hits[stratum],
                    "pct": Rate(hits[stratum], sizes[stratum]).percent if sizes[stratum] else 0.0,
                }
                for stratum in STRATA
            }
        return {"sizes": dict(sizes), "filters": filters}

    def report(self, result):
        sizes = result["sizes"]
        lines = [
            f"{'filter':28s} {'competition (' + str(sizes['competition']) + ')':>18s} "
            f"{'research (' + str(sizes['research']) + ')':>18s}"
        ]
        for name, strata in result["filters"].items():
            cells = [str(Rate(strata[s]["flagged"], strata[s]["n"])) for s in STRATA]
            lines.append(f"{name:28s} {cells[0]:>18s} {cells[1]:>18s}")
        return lines
