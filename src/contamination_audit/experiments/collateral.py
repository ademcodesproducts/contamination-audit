"""How much training data each filter would throw away, and how much of that is noise.

Decontamination removes every training document that matches a benchmark
item. A document that matches only control items is removed for matching text
that cannot be there, so every such removal is unwarranted by construction.
Documents matching measurement items may or may not be contaminated.
"""

from __future__ import annotations

from ..data import DOLCI
from .base import Experiment
from .rates import Rate


class CollateralLoss(Experiment):
    name = "collateral"
    title = "Training documents each filter would remove"

    def run(self):
        split = self.study.split(DOLCI)
        filters = {}
        for published in self.study.filters:
            scan = self.study.scan(published)
            measurement = scan.documents(split.measurement)
            control = scan.documents(split.control)
            filters[published.name] = {
                "measurement_documents": len(measurement),
                "control_documents": len(control),
                "control_only_documents": len(control - measurement),
                "any_documents": len(measurement | control),
            }
        return {"corpus_n": scan.corpus_n, "filters": filters}

    def report(self, result):
        n = result["corpus_n"]
        lines = [
            f"{'filter':28s} {'for measurement':>17s} {'for control':>15s} {'control only':>14s}"
        ]
        for name, row in result["filters"].items():
            cells = [
                str(Rate(row[key], n))
                for key in ("measurement_documents", "control_documents", "control_only_documents")
            ]
            lines.append(f"{name:28s} {cells[0]:>17s} {cells[1]:>15s} {cells[2]:>14s}")
        return lines
