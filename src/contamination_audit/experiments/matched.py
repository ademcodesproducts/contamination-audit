"""The same competitions one year apart.

The false-discovery share is an upper bound only if control and measurement
items attract spurious matches equally often. That is testable directly: AIME
2025 and AIME 2026 are written to the same conventions, and only one of them
can be in the corpus. If a filter flags both at similar rates, most of what it
flags on the older edition is stock phrasing too.
"""

from __future__ import annotations

from ..data import DOLCI
from ..errors import MissingDataError
from ..filters import OPENTHOUGHTS3_NGRAM, S1
from .base import Experiment
from .rates import Rate

# (could be in the corpus, could not be)
COMPETITION_PAIRS = (("aime_2025", "aime_2026"), ("hmmt_feb_2025", "hmmt_feb_2026"))


class MatchedCompetitions(Experiment):
    name = "matched"
    title = "Flag rates on the same competitions one year apart"

    def run(self):
        corpus = self.study.questions(DOLCI)
        by_competition = self.study.benchmarks.matharena_by_competition()

        sides = {}
        for older, newer in COMPETITION_PAIRS:
            sides.update({older: "measure", newer: "control"})
        missing = [competition for competition in sides if competition not in by_competition]
        if missing:
            raise MissingDataError(f"no benchmark items for {', '.join(missing)}")
        groups = [[item.problem for item in by_competition[c]] for c in sides]

        s1_flags = S1.flag_groups(corpus, *groups)
        ot3_flags = OPENTHOUGHTS3_NGRAM.flag_groups(corpus, *groups)
        return {
            competition: {
                "n": len(texts),
                "s1": Rate(len(s1), len(texts)).percent,
                "ot3": Rate(len(ot3), len(texts)).percent,
                "side": side,
            }
            for (competition, side), texts, s1, ot3 in zip(
                sides.items(), groups, s1_flags, ot3_flags, strict=True
            )
        }

    def report(self, result):
        lines = [f"{'competition':16s} {'items':>5s} {'s1':>7s} {'ot3':>7s}"]
        for competition, row in result.items():
            lines.append(
                f"{competition:16s} {row['n']:5d} {row['s1']:6.1f}% {row['ot3']:6.1f}%"
                f"  ({row['side']})"
            )
        return lines
