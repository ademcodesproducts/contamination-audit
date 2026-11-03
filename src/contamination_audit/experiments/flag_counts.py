"""How many benchmark items each published filter flags on the whole of Dolci.

This is the paper's central table. Every control flag is a false positive by
construction, since those items did not exist when the corpus was assembled.
"""

from __future__ import annotations

from ..data import DOLCI
from .base import Experiment
from .rates import Rate


class FlagCounts(Experiment):
    name = "flag_counts"
    title = "Benchmark items flagged by each published filter, on all of Dolci"

    def run(self):
        split = self.study.split(DOLCI)
        filters = {}
        for published in self.study.filters:
            scan = self.study.scan(published)
            filters[published.name] = {
                "provenance": published.provenance.value,
                "measure": sorted(scan.flagged(split.measurement)),
                "control": sorted(scan.flagged(split.control)),
            }

        return {
            "corpus_n": scan.corpus_n,
            "n_measure": len(split.measurement),
            "n_control": len(split.control),
            "filters": filters,
        }

    def report(self, result):
        lines = [f"{'filter':28s} {'measurement':>14s} {'control':>14s}"]
        for name, flagged in result["filters"].items():
            measurement = Rate(len(flagged["measure"]), result["n_measure"])
            control = Rate(len(flagged["control"]), result["n_control"])
            lines.append(f"{name:28s} {str(measurement):>14s} {str(control):>14s}")

        # decon is the fifth filter, run separately because it is a Rust binary.
        decon = self.study.decon_result("decon", result["corpus_n"])
        if decon:
            measurement = Rate(len(decon["measure"]), result["n_measure"])
            control = Rate(len(decon["control"]), result["n_control"])
            lines.append(f"{'decon':28s} {str(measurement):>14s} {str(control):>14s}")
        else:
            lines.append(f"{'decon':28s} not run on this corpus; see `contamination-audit decon`")
        return lines
