"""Recall, measured by planting known contamination.

Control items cannot be in the corpus, which makes them the one place where
contamination can be added with certainty about the answer. Half of them,
chosen at random within each kind of question, are planted as new training
documents in one of several forms; the other half stay clean. A filter's
recall is the share of planted items it flags, and its false-positive rate is
measured, as before, on the items that were not planted.

Every rule decides each document independently, so flagging the corpus plus
the planted documents is the same as flagging each and taking the union. Only
the planted documents need scanning here; the corpus side comes from the
flag_counts and tuning results.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from pathlib import Path

from ..data import DOLCI, BenchmarkItem
from ..filters import QWEN2, S1_WORDS, FuzzyRatio, MatchingRule, PhraseAgreement, SharedNgram
from ..jsonio import read_jsonl
from .base import Experiment
from .phrases import REQUIRED_FRACTIONS
from .rates import SEED, bootstrap_rate
from .strata import kind_of
from .tuning import FUZZY_THRESHOLDS, NGRAM_SIZES

# Each paraphrase style, by the name of its planted form, and where it is saved.
PARAPHRASE_FILES = {
    "paraphrase": "planted_paraphrases.jsonl",
    "paraphrase_free": "planted_paraphrases_free.jsonl",
}
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def retypeset(problem: str) -> str:
    """The same problem as another source might typeset it.

    Swaps the inline and display math delimiters for their alternatives and
    the commonest command synonyms, leaving the mathematics unchanged.
    """
    if "$" in problem:
        text = re.sub(r"\$\$(.+?)\$\$", r"\\[\1\\]", problem, flags=re.S)
        text = re.sub(r"\$(.+?)\$", r"\\(\1\\)", text, flags=re.S)
    else:
        text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", problem, flags=re.S)
        text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.S)
    for old, new in (
        ("\\dfrac", "\\frac"),
        ("\\leq", "\\le"),
        ("\\geq", "\\ge"),
        ("\\cdot", "\\times"),
    ):
        text = text.replace(old, new)
    return text


def stem(problem: str) -> str:
    """The first half of the problem, as when only its setup is quoted."""
    words = problem.split()
    return " ".join(words[: len(words) // 2])


Form = Callable[[BenchmarkItem, str], str]  # (item, unrelated training question) -> document

FORMS: dict[str, Form] = {
    "verbatim": lambda item, _: item.problem,
    "with_answer": lambda item, _: (
        f"{item.problem}\n\nAnswer: {item.answer}" if item.answer else item.problem
    ),
    "retypeset": lambda item, _: retypeset(item.problem),
    "stem": lambda item, _: stem(item.problem),
    "in_conversation": lambda item, other: (
        f"{other}\n\nNext problem. {item.problem}\n{INSTRUCTION}"
    ),
}


def load_paraphrases(directory: Path) -> dict[str, dict[str, str]]:
    """LLM restatements of the control items by planted form, for those generated."""
    return {
        form: {row["id"]: row["paraphrase"] for row in read_jsonl(directory / filename)}
        for form, filename in PARAPHRASE_FILES.items()
        if (directory / filename).exists()
    }


def planted_documents(study) -> tuple[list[BenchmarkItem], list[int], dict[str, list[str]]]:
    """The controls, which of them are planted, and the planted documents in each form.

    Shared with decon's planted run, so both plant exactly the same documents.
    """
    control = list(study.split(DOLCI).control)
    planted = PlantedRecall._choose(control)
    corpus = [q for q in study.questions(DOLCI) if q.strip()]
    fillers = random.Random(SEED + 1).sample(corpus, len(planted))

    forms = dict(FORMS)
    for form, paraphrases in load_paraphrases(study.paths.benchmarks).items():
        forms[form] = lambda item, _, paraphrases=paraphrases: paraphrases[item.id]
    documents = {
        name: [form(control[i], filler) for i, filler in zip(planted, fillers, strict=True)]
        for name, form in forms.items()
    }
    return control, planted, documents


def swept_rules() -> dict[str, MatchingRule]:
    """Every setting the tuning and phrase experiments sweep, by label."""
    rules: dict[str, MatchingRule] = {}
    for n in NGRAM_SIZES:
        rules[f"word n={n}"] = SharedNgram(n, S1_WORDS)
        rules[f"qwen n={n}"] = SharedNgram(n, QWEN2)
    for threshold in FUZZY_THRESHOLDS:
        rules[f"fuzz {threshold}"] = FuzzyRatio(float(threshold))
    for required in REQUIRED_FRACTIONS:
        rules[f"phrase {required:.2f}"] = PhraseAgreement(required_fraction=required)
    return rules


class PlantedRecall(Experiment):
    name = "planted"
    title = "Recall on control items planted into the corpus"

    def run(self):
        control, planted, forms = planted_documents(self.study)
        held_out = [i for i in range(len(control)) if i not in planted]

        rules = {f.name: f.rule for f in self.study.filters} | swept_rules()
        texts = [item.problem for item in control]
        results = {}
        for form_name, documents in forms.items():
            results[form_name] = {}
            for label, rule in rules.items():
                found = {i for i in rule.flag(documents, texts) if i in planted}
                within = {planted.index(i) for i in found}
                results[form_name][label] = {
                    "recall": bootstrap_rate(within, len(planted)),
                    "recall_by_kind": self._by_kind(control, planted, found),
                }

        return {
            "planted": [control[i].id for i in planted],
            "held_out": len(held_out),
            "forms": results,
        }

    @staticmethod
    def _choose(control: list[BenchmarkItem]) -> list[int]:
        """Half of each kind of question, chosen with a fixed seed."""
        rng = random.Random(SEED)
        chosen: list[int] = []
        for kind in sorted({kind_of(item) for item in control}):
            members = [i for i, item in enumerate(control) if kind_of(item) == kind]
            chosen += rng.sample(members, len(members) // 2)
        return sorted(chosen)

    @staticmethod
    def _by_kind(control, planted, found) -> dict:
        by_kind = {}
        for kind in sorted({kind_of(control[i]) for i in planted}):
            members = [i for i in planted if kind_of(control[i]) == kind]
            hits = sum(i in found for i in members)
            by_kind[kind] = {"n": len(members), "flagged": hits, "pct": 100 * hits / len(members)}
        return by_kind

    def report(self, result):
        lines = [f"{len(result['planted'])} items planted, {result['held_out']} held out"]
        for form, rules in result["forms"].items():
            published = [label for label in rules if " " not in label]
            cells = [f"{label} {rules[label]['recall']['pct']:.0f}%" for label in published]
            lines.append(f"{form:16s} " + "  ".join(cells))
        return lines
