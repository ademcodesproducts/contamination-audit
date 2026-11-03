"""Flag rates by kind of question, on both sides of the temporal split.

The pooled rates average over very different questions. The measurement set is
OlymMATH plus competition problems; the control set is competition problems
plus questions derived from research papers, which filters rarely flag. Pooled,
the control rate reflects which MathArena sets happen to exist for 2026 more
than how often a filter misfires on questions like those it is measuring.
Comparing like with like means comparing competition problems on both sides.
"""

from __future__ import annotations

from collections import Counter

from ..data import DOLCI, BenchmarkItem
from .base import Experiment
from .rates import bootstrap_rate


def kind_of(item: BenchmarkItem) -> str:
    if item.benchmark == "olymmath":
        return "olymmath"
    if item.benchmark.startswith(("arxiv", "brokenarxiv")):
        return "research"
    return "competition"


class Strata(Experiment):
    name = "strata"
    title = "Flag rates by kind of question, measurement and control"

    def run(self):
        flag_counts = self.study.load_result("flag_counts")
        split = self.study.split(DOLCI)
        kinds = {
            "measure": [kind_of(item) for item in split.measurement],
            "control": [kind_of(item) for item in split.control],
        }

        filters = {}
        for name, flagged in flag_counts["filters"].items():
            filters[name] = {}
            for group, group_kinds in kinds.items():
                hit = set(flagged[group])
                filters[name][group] = {}
                for kind in sorted(set(group_kinds)):
                    members = [i for i, k in enumerate(group_kinds) if k == kind]
                    within = {j for j, i in enumerate(members) if i in hit}
                    rate = bootstrap_rate(within, len(members))
                    filters[name][group][kind] = {"n": len(members), "flagged": len(within), **rate}
        return {
            "sizes": {group: dict(sorted(Counter(k).items())) for group, k in kinds.items()},
            "filters": filters,
        }

    def report(self, result):
        lines = []
        for name, groups in result["filters"].items():
            cells = [
                f"{group} {kind} {row['flagged']}/{row['n']} {row['pct']:.1f}% "
                f"[{row['ci'][0]:.1f}, {row['ci'][1]:.1f}]"
                for group, kinds in groups.items()
                for kind, row in kinds.items()
            ]
            lines.append(f"{name}: " + ";  ".join(cells))
        return lines
