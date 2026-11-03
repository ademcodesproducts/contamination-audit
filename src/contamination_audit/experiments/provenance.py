"""When the research behind each arXiv-derived control first appeared.

A control question is published after the corpus, but the result it is built
from may not be: the question can come from a paper whose first preprint
predates the corpus, in which case the content could be in it. This compares
the date each source paper first appeared on arXiv with the corpus's release.
Controls whose paper appeared on or before that day are dropped from the
corpus's split; this records how many.
"""

from __future__ import annotations

from collections import Counter

from ..data import DOLCI, NEMOTRON, TULU
from .base import Experiment
from .strata import kind_of


class ControlProvenance(Experiment):
    name = "control_dates"
    title = "First arXiv appearance of the papers behind each research-derived control"

    def run(self):
        result = {}
        for corpus in (DOLCI, TULU, NEMOTRON):
            candidates = [
                item
                for item in corpus.split_rule(self.study.benchmarks).control
                if item.first_posted is not None
            ]
            dropped = [item for item in candidates if item.first_posted <= corpus.released]
            earliest = min((item.first_posted for item in candidates), default=None)
            result[corpus.name] = {
                "released": corpus.released.isoformat(),
                "arxiv_controls": len(candidates),
                # Research questions whose source paper could not be dated stay in;
                # this should be zero.
                "undated_research_controls": sum(
                    item.first_posted is None and kind_of(item) == "research"
                    for item in corpus.split_rule(self.study.benchmarks).control
                ),
                "first_posted_by_month": dict(
                    sorted(Counter(i.first_posted.strftime("%Y-%m") for i in candidates).items())
                ),
                "earliest": earliest.isoformat() if earliest else None,
                "dropped_on_or_before_release": sorted(item.id for item in dropped),
            }
        return result

    def report(self, result):
        return [
            f"{corpus}: {row['arxiv_controls']} arXiv-derived controls, earliest paper "
            f"{row['earliest']}, released {row['released']}; "
            f"{len(row['dropped_on_or_before_release'])} dropped as not postdating it"
            for corpus, row in result.items()
        ]
