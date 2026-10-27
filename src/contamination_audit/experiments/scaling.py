"""How false positives grow with the size of the corpus.

Searching more documents gives a filter more chances to find matching text, so
flag rates on both benchmark groups rise with corpus size. The question is
whether the share of flags that are false positives rises too. Each size is a
prefix of the next, so the curve varies corpus size and nothing else.
"""

from __future__ import annotations

from ..data import DOLCI
from ..filters import OPENTHOUGHTS3_NGRAM, OPENTHOUGHTS_114K, S1
from .base import Experiment
from .rates import Rate, false_discovery_share, format_share

CORPUS_SIZES = (12_500, 25_000, 50_000, 100_000, 150_000, 250_000, 390_000)


class Scaling(Experiment):
    name = "scaling"
    title = "Flag rates of s1 and the OpenThoughts3 n-gram as the corpus grows"

    def run(self):
        split = self.study.split(DOLCI)
        largest = self.study.questions(DOLCI, CORPUS_SIZES[-1])
        groups = (split.measurement_texts, split.control_texts)

        rows = []
        for size in CORPUS_SIZES:
            corpus = largest[:size]
            row = {"corpus": size}
            for prefix, rule in (("s1", S1.rule), ("ot3", OPENTHOUGHTS3_NGRAM)):
                measurement, control = (
                    Rate(len(flagged), len(group))
                    for flagged, group in zip(
                        rule.flag_groups(corpus, *groups), groups, strict=True
                    )
                )
                row[f"{prefix}_det"] = measurement.percent
                row[f"{prefix}_fp"] = control.percent
                row[f"{prefix}_frac"] = false_discovery_share(control, measurement)
            # OpenThoughts-114K's whole-string filter, as a check that it stays silent.
            row["fuzz"] = sum(len(f) for f in OPENTHOUGHTS_114K.flag_groups(corpus, *groups))
            rows.append(row)
        return rows

    def report(self, result):
        lines = [
            f"{'documents':>10s} | {'s1 flagged':>10s} {'false pos':>9s} {'share':>6s} | "
            f"{'ot3 flagged':>11s} {'false pos':>9s} {'share':>6s}"
        ]
        for row in result:
            lines.append(
                f"{row['corpus']:10,d} | {row['s1_det']:9.1f}% {row['s1_fp']:8.1f}% "
                f"{format_share(row['s1_frac']):>6s} | {row['ot3_det']:10.1f}% "
                f"{row['ot3_fp']:8.1f}% {format_share(row['ot3_frac']):>6s}"
            )
        return lines
