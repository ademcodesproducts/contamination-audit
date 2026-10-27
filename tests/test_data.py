"""Benchmarks, corpora, and the temporal split each corpus defines."""

import pytest

from contamination_audit.data import DOLCI, NEMOTRON, TULU, BenchmarkLibrary
from contamination_audit.errors import MissingDataError


@pytest.fixture
def library(tiny_root):
    return BenchmarkLibrary.load(tiny_root / "data" / "benchmarks")


def ids(items):
    return [item.id for item in items]


def test_olymmath_items_are_labelled_as_such(library):
    assert {item.benchmark for item in library.olymmath} == {"olymmath"}


def test_dolci_split_is_olymmath_and_matharena_against_2026(library):
    split = DOLCI.split(library)
    assert ids(split.measurement) == [
        "olym_en_hard_0000",
        "olym_en_easy_0000",
        "aime_2024_I_0000",
        "aime_2025_0000",
        "hmmt_feb_2025_0000",
    ]
    assert ids(split.control) == ["aime_2026_0000", "hmmt_feb_2026_0000", "arxivmath-0126_0000"]


def test_tulu_split_measures_only_on_2024(library):
    # Tulu 3 shipped in November 2024, so everything later becomes a control,
    # in the same order as before.
    split = TULU.split(library)
    assert ids(split.measurement) == ["aime_2024_I_0000"]
    assert ids(split.control) == [
        "olym_en_hard_0000",
        "olym_en_easy_0000",
        "aime_2025_0000",
        "hmmt_feb_2025_0000",
        "aime_2026_0000",
        "hmmt_feb_2026_0000",
        "arxivmath-0126_0000",
    ]


def test_nemotron_split_moves_early_2026_into_measurement(library):
    split = NEMOTRON.split(library)
    assert ids(split.measurement)[-3:] == [
        "aime_2026_0000",
        "hmmt_feb_2026_0000",
        "arxivmath-0126_0000",
    ]
    assert ids(split.control) == ["arxivmath-0826_0000"]


def test_corpus_reads_a_prefix(tiny_root):
    documents = DOLCI.read(tiny_root / "data" / "corpora", 2)
    assert [d.train_id for d in documents] == ["doc0", "doc1"]


def test_a_short_corpus_is_an_error_not_a_smaller_result(tiny_root):
    with pytest.raises(MissingDataError, match="holds 4 documents but 5"):
        DOLCI.read(tiny_root / "data" / "corpora", 5)


def test_a_missing_benchmark_file_says_how_to_fetch_it(tmp_path):
    with pytest.raises(MissingDataError, match="contamination-audit fetch"):
        BenchmarkLibrary.load(tmp_path)
