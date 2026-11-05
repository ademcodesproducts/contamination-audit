"""A string filter and an embedding filter together, chosen and tested on separate items.

The two families fail on different copies: string rules catch a problem inside
a longer prompt and miss a restatement, and embeddings do the reverse. This
flags an item when either rule does. Every setting of each family, and every
pair, is scored on the tuning half of the controls. The one kept has the most
recall among those within a false-positive budget fixed in advance, and only the
test half is reported, beside the best single family chosen the same way.

Recall counts an item when a planted document flags it, as in `planted`.
False positives are read on clean controls over all of Dolci.
"""

from __future__ import annotations

import random

from ..data import DOLCI
from ..filters.embedding import Encoder
from .base import Experiment
from .embedding import THRESHOLDS
from .planted import FORMS, load_paraphrases, swept_rules
from .rare_ngrams import split_items
from .rates import SEED
from .strata import kind_of

FP_BUDGET = 2.0  # percent of the tuning half's clean controls, fixed in advance

STRING_RULES = (
    "word n=8",
    "word n=10",
    "word n=13",
    "word n=15",
    "qwen n=10",
    "qwen n=13",
    "qwen n=15",
    "qwen n=20",
    "fuzz 95",
    "fuzz 75",
    "fuzz 70",
    "fuzz 65",
    "phrase 0.20",
    "phrase 0.10",
)


def best_under_budget(scores: dict, budget: float = FP_BUDGET):
    """The key with the most recall among those within the false-positive budget.

    Ties go to fewer false positives, then to the earliest key. Returns None if
    nothing is within budget.
    """
    allowed = {key: score for key, score in scores.items() if score["fp"] <= budget}
    if not allowed:
        return None
    return max(allowed, key=lambda key: (allowed[key]["recall"], -allowed[key]["fp"]))


def percent(flagged: set[int], members: list[int]) -> float:
    return 100 * sum(i in flagged for i in members) / len(members)


class CombinedFilter(Experiment):
    name = "combined"
    title = "A string filter and an embedding filter together, tuned and tested on separate items"

    def run(self):
        split = self.study.split(DOLCI)
        control = list(split.control)
        n_measure = len(split.measurement)
        cells = split_items([kind_of(item) for item in control], random.Random(SEED))
        planted = cells["tune_planted"] + cells["test_planted"]

        corpus = self.study.full_questions(DOLCI)
        fillers = random.Random(SEED + 1).sample([q for q in corpus if q.strip()], len(planted))
        forms = {
            name: [form(control[i], f) for i, f in zip(planted, fillers, strict=True)]
            for name, form in FORMS.items()
        }
        for form, paraphrases in load_paraphrases(self.study.paths.benchmarks).items():
            forms[form] = [paraphrases[control[i].id] for i in planted]

        # Flags over the corpus, as positions in measurement + control.
        tuning = self.study.load_result("tuning")
        string_corpus: dict[str, set[int]] = {}
        for block, prefix in (("word", "word n="), ("qwen", "qwen n=")):
            for row in tuning[block]:
                string_corpus[f"{prefix}{row['n']}"] = set(row["flagged"])
        for row in tuning["fuzz"]:
            string_corpus[f"fuzz {row['threshold']}"] = set(row["flagged"])
        for row in self.study.load_result("phrases"):
            string_corpus[f"phrase {row['required']:.2f}"] = set(row["flagged"])

        nearest = self.study.load_result("embedding")["nearest"]
        items = list(split.measurement) + control
        embedding_corpus = {
            t: {i for i, item in enumerate(items) if nearest[item.id][0] >= t} for t in THRESHOLDS
        }

        # Flags from the planted documents, as positions in `control`.
        rules = swept_rules()
        texts = [item.problem for item in control]
        encoder = Encoder()
        planted_vectors = encoder.encode([control[i].problem for i in planted])
        string_planted = {form: {} for form in forms}
        embedding_planted = {form: {} for form in forms}
        for form, documents in forms.items():
            for label in STRING_RULES:
                string_planted[form][label] = rules[label].flag(documents, texts) & set(planted)
            best = (encoder.encode(documents) @ planted_vectors.T).max(axis=0)
            for t in THRESHOLDS:
                embedding_planted[form][t] = {
                    planted[k] for k, similarity in enumerate(best) if similarity >= t
                }

        def clean_flags(label, t):
            flagged = set()
            if label:
                flagged |= string_corpus[label]
            if t:
                flagged |= embedding_corpus[t]
            return flagged

        def control_flags(label, t):
            return {i - n_measure for i in clean_flags(label, t) if i >= n_measure}

        def planted_flags(form, label, t):
            flagged = set()
            if label:
                flagged |= string_planted[form][label]
            if t:
                flagged |= embedding_planted[form][t]
            return flagged

        def score(label, t, half):
            recalls = {
                form: percent(planted_flags(form, label, t), cells[f"{half}_planted"])
                for form in forms
            }
            return {
                "fp": percent(control_flags(label, t), cells[f"{half}_clean"]),
                "recall": sum(recalls.values()) / len(recalls),
                "forms": recalls,
            }

        string_only = {(label, None): score(label, None, "tune") for label in STRING_RULES}
        embedding_only = {(None, t): score(None, t, "tune") for t in THRESHOLDS}
        both = {(label, t): score(label, t, "tune") for label in STRING_RULES for t in THRESHOLDS}

        selected = {}
        for name, scores in (
            ("string", string_only),
            ("embedding", embedding_only),
            ("combined", both),
        ):
            key = best_under_budget(scores)
            if key is None:
                selected[name] = None
                continue
            label, t = key
            flagged = clean_flags(label, t)
            selected[name] = {
                "string_rule": label,
                "cosine": t,
                "tune": scores[key],
                "test": score(label, t, "test"),
                "measurement_flagged": sum(i < n_measure for i in flagged),
            }
        return {
            "budget": FP_BUDGET,
            "cells": {cell: len(members) for cell, members in cells.items()},
            "n_measure": n_measure,
            "selected": selected,
        }

    def report(self, result):
        lines = [f"chosen on the tuning half, false positives at most {result['budget']}%"]
        for name, row in result["selected"].items():
            if row is None:
                lines.append(f"{name:10s} nothing within budget")
                continue
            test = row["test"]
            cells = " ".join(f"{form[:10]} {pct:.0f}" for form, pct in test["forms"].items())
            lines.append(
                f"{name:10s} {row['string_rule'] or '-'} + {row['cosine'] or '-'}: "
                f"test false pos {test['fp']:.1f}%, mean recall {test['recall']:.1f}%, "
                f"tune {row['tune']['fp']:.1f}%/{row['tune']['recall']:.1f}%; {cells}"
            )
        return lines
