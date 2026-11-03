"""Planting forms and the tune/test split for the rare-n-gram filter."""

import random

from contamination_audit.data import BenchmarkItem
from contamination_audit.experiments.planted import FORMS, retypeset, stem
from contamination_audit.experiments.rare_ngrams import split_items

ITEM = BenchmarkItem("x", "aime_2026", r"Let $x^2 = 4$ and $$y \leq 3$$. Find $x + y$.", "5")


def test_retypeset_changes_delimiters_not_mathematics():
    assert retypeset(ITEM.problem) == r"Let \(x^2 = 4\) and \[y \le 3\]. Find \(x + y\)."


def test_retypeset_goes_both_ways():
    assert retypeset(r"Find \(x\) where \[x = 1\].") == r"Find $x$ where $$x = 1$$."


def test_stem_keeps_the_first_half_of_the_words():
    assert stem("one two three four five") == "one two"


def test_every_form_contains_the_start_of_the_problem():
    for name, form in FORMS.items():
        document = form(ITEM, "an unrelated training question")
        assert "Let" in document, name


def test_tune_and_test_cells_partition_each_kind():
    kinds = ["competition"] * 8 + ["research"] * 12
    cells = split_items(kinds, random.Random(0))
    assert sorted(i for members in cells.values() for i in members) == list(range(20))
    assert all(len(members) == 5 for members in cells.values())
