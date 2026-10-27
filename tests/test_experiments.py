"""Experiments end to end, on the miniature study in conftest.py."""

import json

from contamination_audit.experiments import EXPERIMENTS
from contamination_audit.experiments.agreement import FilterAgreement
from contamination_audit.experiments.flag_counts import FlagCounts
from contamination_audit.experiments.mechanism import Mechanism
from contamination_audit.experiments.phrases import REQUIRED_FRACTIONS, PhraseAgreementSweep
from contamination_audit.experiments.strata import ControlStrata
from contamination_audit.experiments.validation import ControlValidation


def test_every_experiment_has_a_unique_name_and_a_title():
    assert len(EXPERIMENTS) == 12
    assert all(name == cls.name and cls.title for name, cls in EXPERIMENTS.items())


def test_experiments_that_read_flag_counts_run_after_it():
    order = list(EXPERIMENTS)
    assert order.index("flag_counts") < order.index("agreement")
    assert order.index("flag_counts") < order.index("strata")


def test_flag_counts(tiny_study):
    result = tiny_study.run(FlagCounts)
    assert (result["n_measure"], result["n_control"]) == (5, 3)
    # The verbatim document and the stock phrase both trigger the 8-gram filter;
    # only the verbatim one clears a 95 similarity.
    assert result["filters"]["words-8gram"] == {
        "provenance": "code",
        "measure": [0, 3],
        "control": [0],
    }
    assert result["filters"]["fuzzy-95"] == {"provenance": "prose", "measure": [0], "control": []}


def test_results_are_saved_as_json(tiny_study):
    tiny_study.run(FlagCounts)
    saved = json.loads(tiny_study.result_path("flag_counts").read_text(encoding="utf-8"))
    assert saved["filters"]["words-8gram"]["control"] == [0]


def test_strata(tiny_study):
    tiny_study.run(FlagCounts)
    result = tiny_study.run(ControlStrata)
    assert result["sizes"] == {"competition": 2, "research": 1}
    assert result["filters"]["words-8gram"]["competition"] == {"n": 2, "flagged": 1, "pct": 50.0}


def test_agreement(tiny_study):
    tiny_study.run(FlagCounts)
    result = tiny_study.run(FilterAgreement)
    # Item 0 is flagged by both cheap filters, item 3 by one; decon flags none.
    assert result["flagged_any"] == 2
    assert result["consensus"][1]["items"] == 1
    assert result["consensus"][2]["items"] == 1
    assert result["jaccard"] == {"words-8gram|fuzzy-95": 0.5}


def test_mechanism_traces_the_false_positive_to_the_stock_phrase(tiny_study):
    result = tiny_study.run(Mechanism)
    assert result["false_positives"] == 1
    assert all(
        row["ngram"] in "let m and n be relatively prime positive integers find m plus n"
        for row in result["top"]
    )


def test_validation_finds_the_longest_shared_spans(tiny_study):
    result = tiny_study.run(ControlValidation)
    # The 13-word stock phrase outranks the 10-word verbatim problem.
    measurement = [(row["item"], row["n"]) for row in result["longest"]["measurement"]]
    assert measurement == [("aime_2025_0000", 13), ("olym_en_hard_0000", 10)]
    assert result["longest"]["control"][0]["benchmark"] == "aime_2026"
    assert result["distribution"]["control"] == {"0": 2, "13": 1}


def test_phrase_sweep_reports_every_setting(tiny_study):
    result = tiny_study.run(PhraseAgreementSweep)
    assert [row["required"] for row in result] == list(REQUIRED_FRACTIONS)
