"""Does requiring agreement across an item suppress stock-phrase matches?

Most false positives come from one conventional phrase shared with an unrelated
training question. This sweeps how many of an item's twenty sampled phrases a
single document must contain before the item is flagged.
"""

from __future__ import annotations

from ..data import DOLCI
from ..filters import PhraseAgreement
from .base import Experiment
from .rates import Rate, false_discovery_share, format_share

REQUIRED_FRACTIONS = (0.50, 0.30, 0.20, 0.10, 0.05)


class PhraseAgreementSweep(Experiment):
    name = "phrases"
    title = "Phrase-agreement filtering: 20 sampled 8-word windows per item"

    def run(self):
        corpus = self.study.questions(DOLCI)
        split = self.study.split(DOLCI)

        rows = []
        for required in REQUIRED_FRACTIONS:
            rule = PhraseAgreement(required_fraction=required)
            flagged_measurement, flagged_control = rule.flag_groups(
                corpus, split.measurement_texts, split.control_texts
            )
            measurement = Rate(len(flagged_measurement), len(split.measurement))
            control = Rate(len(flagged_control), len(split.control))
            rows.append(
                {
                    "required": required,
                    "detect": measurement.percent,
                    "fp": control.percent,
                    "frac": false_discovery_share(control, measurement),
                }
            )
        return rows

    def report(self, result):
        lines = [f"{'required':>9s} {'flagged':>9s} {'false pos':>10s} {'share':>6s}"]
        for row in result:
            lines.append(
                f"{row['required']:9.2f} {row['detect']:8.1f}% {row['fp']:9.1f}% "
                f"{format_share(row['frac']):>6s}"
            )
        return lines
