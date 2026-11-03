"""How false positives grow with the size of the corpus.

Searching more documents gives a filter more chances to find matching text, so
flag rates on both benchmark groups rise with corpus size. The question is
whether the share of flags that are false positives rises too.

Each size is a prefix of the corpus, so every curve varies size and nothing
else, ending at the whole of Dolci. A filter flags an item when any document
matches it, so the flags at a size are read off the full-corpus scan by
counting only documents below that index; no prefix is scanned twice.
"""

from __future__ import annotations

from ..data import DOLCI
from ..filters import OPENTHOUGHTS3, OPENTHOUGHTS_114K, S1
from .base import Experiment
from .rates import Rate, false_discovery_share, format_share

# The last point is the whole corpus, whatever its size.
CORPUS_SIZES = (12_500, 25_000, 50_000, 100_000, 150_000, 250_000, 390_000, 1_000_000, None)


class Scaling(Experiment):
    name = "scaling"
    title = "Flag rates of s1 and OpenThoughts3 as the corpus grows, up to all of Dolci"

    def run(self):
        split = self.study.split(DOLCI)
        scans = {"s1": self.study.scan(S1), "ot3": self.study.scan(OPENTHOUGHTS3)}
        fuzzy = self.study.scan(OPENTHOUGHTS_114K)

        rows = []
        for size in CORPUS_SIZES:
            size = size or fuzzy.corpus_n
            row = {"corpus": size}
            for prefix, scan in scans.items():
                measurement = Rate(
                    len(scan.flagged(split.measurement, within=size)), len(split.measurement)
                )
                control = Rate(len(scan.flagged(split.control, within=size)), len(split.control))
                row[f"{prefix}_det"] = measurement.percent
                row[f"{prefix}_fp"] = control.percent
                row[f"{prefix}_frac"] = false_discovery_share(control, measurement)
            # OpenThoughts-114K's whole-string filter, as a check that it stays silent.
            row["fuzz"] = len(fuzzy.flagged(split.measurement, within=size)) + len(
                fuzzy.flagged(split.control, within=size)
            )
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
