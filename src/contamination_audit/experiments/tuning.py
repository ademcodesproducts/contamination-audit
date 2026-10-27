"""Tuning a filter's setting against the control set, without labels.

Each published filter ships one setting: an n-gram length or a similarity
cutoff. Sweeping that setting and reading the control set at each point traces
how much a filter flags against how often it is provably wrong.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..data import DOLCI, TemporalSplit
from ..filters import QWEN2, S1_WORDS, Tokenizer, ngrams
from ..filters.rules import best_fuzzy_scores
from .base import Experiment
from .rates import Rate, false_discovery_share

NGRAM_SIZES = (5, 8, 10, 13, 15, 20, 25, 30)
FUZZY_THRESHOLDS = (95, 90, 85, 80, 75, 70, 65, 60)


def flag_by_ngram_size(
    corpus: Sequence[str], benchmark: Sequence[str], sizes: Sequence[int], tokenizer: Tokenizer
) -> dict[int, set[int]]:
    """`SharedNgram(n, tokenizer).flag` for every n in `sizes`, in one pass.

    Tokenizing the corpus dominates the cost, so this tokenizes each document
    once and checks every n-gram length against it.
    """
    items_with: dict[int, dict[tuple, set[int]]] = {n: {} for n in sizes}
    for item, text in enumerate(benchmark):
        tokens = tokenizer(text)
        for n in sizes:
            for gram in ngrams(tokens, n):
                items_with[n].setdefault(gram, set()).add(item)

    flagged: dict[int, set[int]] = {n: set() for n in sizes}
    for document in corpus:
        if not document.strip():
            continue
        tokens = tokenizer(document)
        for n in sizes:
            index = items_with[n]
            for gram in ngrams(tokens, n):
                hit = index.get(gram)
                if hit:
                    flagged[n] |= hit
    return flagged


class Tuning(Experiment):
    name = "tuning"
    title = "Measurement against control flag rates as each filter's setting varies"

    def run(self):
        corpus = self.study.questions(DOLCI)
        split = self.study.split(DOLCI)
        return {
            "word": self._ngram_sweep(corpus, split, S1_WORDS),
            "qwen": self._ngram_sweep(corpus, split, QWEN2),
            "fuzz": self._fuzzy_sweep(corpus, split),
        }

    def _ngram_sweep(self, corpus, split, tokenizer):
        benchmark = split.measurement_texts + split.control_texts
        flagged = flag_by_ngram_size(corpus, benchmark, NGRAM_SIZES, tokenizer)
        return [self._row("n", n, flagged[n], split) for n in NGRAM_SIZES]

    def _fuzzy_sweep(self, corpus, split):
        # Whole-string similarity does not depend on the threshold, so one score
        # matrix answers every threshold at once.
        best = best_fuzzy_scores(corpus, split.measurement_texts + split.control_texts, floor=50)
        return [
            self._row("threshold", t, set((best >= t).nonzero()[0].tolist()), split)
            for t in FUZZY_THRESHOLDS
        ]

    @staticmethod
    def _row(setting: str, value, flagged: set[int], split: TemporalSplit) -> dict:
        n_measure = len(split.measurement)
        measurement = Rate(sum(1 for i in flagged if i < n_measure), n_measure)
        control = Rate(sum(1 for i in flagged if i >= n_measure), len(split.control))
        return {
            setting: value,
            "detect": measurement.percent,
            "fp": control.percent,
            "frac": false_discovery_share(control, measurement),
        }

    def report(self, result):
        lines = []
        for block, setting in (("word", "n"), ("qwen", "n"), ("fuzz", "threshold")):
            lines.append(f"{block}: {setting:>9s} {'flagged':>8s} {'false pos':>9s}")
            for row in result[block]:
                lines.append(f"      {row[setting]:9} {row['detect']:7.1f}% {row['fp']:8.1f}%")
        return lines
