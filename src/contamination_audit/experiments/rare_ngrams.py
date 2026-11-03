"""Evaluating the rare-n-gram filter, with its cutoff chosen on separate items.

The filter ignores 8-grams found in more than k training documents. Choosing k
and reporting it on the same items would repeat the mistake the paper warns
against, so the control items are split, within each kind of question, into a
tuning half and a test half. Within each half, some items are planted into the
corpus and the rest stay clean. k is chosen on the tuning half and the filter
is reported on the test half, beside s1's unfiltered 8-gram rule.

Planting five copies of each item checks the obvious failure: a leaked problem
duplicated across datasets is itself frequent, and could be mistaken for
convention.
"""

from __future__ import annotations

import random
from collections import Counter

from ..data import DOLCI
from ..filters import S1_WORDS, ngrams
from ..filters.rules import shared_ngrams
from .base import Experiment
from .planted import FORMS, load_paraphrases
from .rates import SEED
from .strata import kind_of

N = 8
CUTOFFS = (1, 2, 3, 5, 10, 20, 50, 100)
COPIES = 5


def split_items(kinds: list[str], rng: random.Random) -> dict[str, list[int]]:
    """Tune and test halves, each split into planted and clean, within each kind."""
    cells: dict[str, list[int]] = {
        "tune_planted": [],
        "tune_clean": [],
        "test_planted": [],
        "test_clean": [],
    }
    for kind in sorted(set(kinds)):
        members = [i for i, k in enumerate(kinds) if k == kind]
        rng.shuffle(members)
        quarter = len(members) // 4
        for position, cell in enumerate(cells):
            stop = None if position == 3 else (position + 1) * quarter
            cells[cell] += members[position * quarter : stop]
    return {cell: sorted(members) for cell, members in cells.items()}


class RareNgramEvaluation(Experiment):
    name = "rare_ngrams"
    title = "The rare-n-gram filter on all of Dolci, tuned and tested on separate items"

    def run(self):
        corpus = self.study.full_questions(DOLCI)
        control = list(self.study.split(DOLCI).control)
        cells = split_items([kind_of(item) for item in control], random.Random(SEED))
        planted = cells["tune_planted"] + cells["test_planted"]

        items_with: dict[tuple, set[int]] = {}
        for item, entry in enumerate(control):
            for gram in ngrams(S1_WORDS(entry.problem), N):
                items_with.setdefault(gram, set()).add(item)
        in_corpus = shared_ngrams(corpus, items_with, N, S1_WORDS)
        corpus_frequency = Counter(gram for grams in in_corpus.values() for gram in grams)

        fillers = random.Random(SEED + 1).sample([q for q in corpus if q.strip()], len(planted))
        paraphrases = load_paraphrases(self.study.paths.benchmarks)
        forms = {
            name: [form(control[i], f) for i, f in zip(planted, fillers, strict=True)]
            for name, form in FORMS.items()
        }
        forms["five_copies"] = [control[i].problem for i in planted for _ in range(COPIES)]
        if paraphrases:
            forms["paraphrase"] = [paraphrases[control[i].id] for i in planted]

        results = {}
        for form, documents in forms.items():
            in_planted = shared_ngrams(documents, items_with, N, S1_WORDS)
            frequency = corpus_frequency + Counter(
                g for grams in in_planted.values() for g in grams
            )
            results[form] = {}
            for cutoff in (*CUTOFFS, None):
                flagged = set()
                for grams in [*in_corpus.values(), *in_planted.values()]:
                    for gram in grams:
                        if cutoff is None or frequency[gram] <= cutoff:
                            flagged |= items_with[gram]
                results[form]["s1" if cutoff is None else str(cutoff)] = {
                    cell: 100 * sum(i in flagged for i in members) / len(members)
                    for cell, members in cells.items()
                }

        chosen = max(CUTOFFS, key=lambda k: self._tuning_score(results, str(k)))
        return {
            "cells": {cell: len(members) for cell, members in cells.items()},
            "chosen_cutoff": chosen,
            "forms": results,
        }

    @staticmethod
    def _tuning_score(results: dict, cutoff: str) -> float:
        """Mean over planting forms of recall minus false positives, on the tuning half."""
        rows = [form[cutoff] for form in results.values()]
        return sum(row["tune_planted"] - row["tune_clean"] for row in rows) / len(rows)

    def report(self, result):
        chosen = str(result["chosen_cutoff"])
        lines = [
            f"cutoff chosen on the tuning half: at most {chosen} documents per 8-gram",
            f"{'test half':16s} {'recall':>14s} {'false pos':>14s}   (rare n-grams vs s1)",
        ]
        for form, rows in result["forms"].items():
            rare, s1 = rows[chosen], rows["s1"]
            lines.append(
                f"{form:16s} {rare['test_planted']:5.1f}% vs {s1['test_planted']:5.1f}%"
                f" {rare['test_clean']:5.1f}% vs {s1['test_clean']:5.1f}%"
            )
        return lines
