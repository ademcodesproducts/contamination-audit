"""The main comparison, repeated on two further corpora.

If false positives are a property of the filters rather than of Dolci, they
should reappear elsewhere. Each corpus gets its own temporal split, derived
from its own release date, so both the measurement and control sets change
with the corpus.
"""

from __future__ import annotations

from typing import ClassVar

from ..data import NEMOTRON, TULU, TrainingCorpus
from .base import Experiment
from .rates import Rate, false_discovery_share


class Replication(Experiment):
    corpus: ClassVar[TrainingCorpus]

    def run(self):
        documents = self.study.questions(self.corpus)
        split = self.study.split(self.corpus)

        filters = {}
        for published in self.study.filters:
            flagged_measurement, flagged_control = published.flag_groups(
                documents, split.measurement_texts, split.control_texts
            )
            measurement = Rate(len(flagged_measurement), len(split.measurement))
            control = Rate(len(flagged_control), len(split.control))
            filters[published.name] = {
                "detect": measurement.percent,
                "fp": control.percent,
                "n_detect": measurement.count,
                "n_fp": control.count,
                "fd_share": false_discovery_share(control, measurement),
            }

        return {
            "corpus": self.corpus.name,
            "docs": len(documents),
            "n_measure": len(split.measurement),
            "n_control": len(split.control),
            "filters": filters,
        }

    def report(self, result):
        lines = [
            f"measurement {result['n_measure']} items, control {result['n_control']} items, "
            f"released {self.corpus.released:%B %Y}",
            f"{'filter':28s} {'measurement':>14s} {'control':>14s}",
        ]
        for name, row in result["filters"].items():
            measurement = Rate(row["n_detect"], result["n_measure"])
            control = Rate(row["n_fp"], result["n_control"])
            lines.append(f"{name:28s} {str(measurement):>14s} {str(control):>14s}")
        return lines


class TuluReplication(Replication):
    """Tulu 3 shipped in November 2024, so every 2025 and 2026 item is a control.

    That leaves only the 2024 competitions to measure on, a small set. The
    quantity of interest here is the false-positive rate, though, and for that
    the control side is what matters, and it is large.
    """

    name = "tulu"
    title = "Replication on the Tulu 3 math SFT data (100,000 documents)"
    corpus = TULU


class NemotronReplication(Replication):
    """Nemotron's May 2026 release falls between Dolci's two groups.

    Everything published before May 2026 could be inside it and becomes
    measurement, including the early-2026 sets that are controls for Dolci.
    Only sets from June 2026 onward can serve as its control.
    """

    name = "nemotron"
    title = "Replication on Nemotron-SFT-Math-v4 (100,000 documents)"
    corpus = NEMOTRON
