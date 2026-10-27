"""How many benchmark items each published filter flags on the main corpus.

This is the paper's central table. Every control flag is a false positive by
construction, since those items did not exist when the corpus was assembled.
"""

from __future__ import annotations

from ..data import DOLCI
from .base import Experiment
from .rates import Rate


class FlagCounts(Experiment):
    name = "flag_counts"
    title = "Benchmark items flagged by each published filter (100,000 Dolci documents)"

    def run(self):
        corpus = self.study.questions(DOLCI)
        split = self.study.split(DOLCI)

        filters = {}
        for published in self.study.filters:
            measurement, control = published.flag_groups(
                corpus, split.measurement_texts, split.control_texts
            )
            filters[published.name] = {
                "provenance": published.provenance.value,
                "measure": sorted(measurement),
                "control": sorted(control),
            }

        return {
            "corpus_n": len(corpus),
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
        if self.study.has_result("decon"):
            decon = self.study.load_result("decon")
            measurement = Rate(decon["measure"], result["n_measure"])
            control = Rate(decon["control"], result["n_control"])
            lines.append(f"{'decon':28s} {str(measurement):>14s} {str(control):>14s}")
        else:
            lines.append(f"{'decon':28s} not run; see `contamination-audit decon`")
        return lines
