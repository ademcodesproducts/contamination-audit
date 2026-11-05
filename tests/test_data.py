"""Benchmarks, corpora, and the temporal split each corpus defines."""

import pytest

from contamination_audit.data import DOLCI, NEMOTRON, TULU, BenchmarkLibrary
from contamination_audit.data.download import arxiv_id
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
    # Tulu 3 shipped in November 2024, so every later competition becomes a control,
    # in the same order as before. OlymMATH is sourced from printed publications, so
    # it is neither: its 2025 date does not date its problems.
    split = TULU.split(library)
    assert ids(split.measurement) == ["aime_2024_I_0000"]
    assert ids(split.control) == [
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


def test_the_whole_corpus_needs_a_finished_download(tiny_root):
    corpora = tiny_root / "data" / "corpora"
    assert len(DOLCI.read(corpora, None)) == 4

    with open(corpora / "dolci.jsonl", "a", encoding="utf-8") as extra:
        extra.write('{"train_id": "doc4", "problem": "one more", "source": "a"}\n')
    with pytest.raises(MissingDataError, match="its download wrote 4"):
        DOLCI.read(corpora, None)

    (corpora / "dolci.complete.json").unlink()
    with pytest.raises(MissingDataError, match="not a complete download"):
        DOLCI.read(corpora, None)


def test_a_short_corpus_is_an_error_not_a_smaller_result(tiny_root):
    with pytest.raises(MissingDataError, match="holds 4 documents but 5"):
        DOLCI.read(tiny_root / "data" / "corpora", 5)


def test_a_missing_benchmark_file_says_how_to_fetch_it(tmp_path):
    with pytest.raises(MissingDataError, match="contamination-audit fetch"):
        BenchmarkLibrary.load(tmp_path)


def test_arxiv_ids_stored_as_numbers_get_their_trailing_zero_back():
    assert arxiv_id(2601.0195) == "2601.01950"
    assert arxiv_id("2603.00693") == "2603.00693"


def test_controls_whose_source_paper_predates_the_corpus_are_dropped(library):
    from datetime import date

    from contamination_audit.data import TrainingCorpus

    late = TrainingCorpus("late", "x.jsonl", date(2026, 1, 15), BenchmarkLibrary.dolci_split)
    dated = library.with_dates({"2601.00001": "2026-01-10"})
    research = next(i for i in dated.control if i.benchmark.startswith("arxiv"))
    assert research.first_posted is None  # the fixture's item has no source
    sourced = dated.__class__(
        olymmath=dated.olymmath,
        measurement=dated.measurement,
        control=tuple(
            i.__class__(i.id, i.benchmark, i.problem, i.answer, "2601.00001")
            if i is research
            else i
            for i in dated.control
        ),
        late_control=dated.late_control,
        earlier_editions=dated.earlier_editions,
    ).with_dates({"2601.00001": "2026-01-10"})
    assert research.id not in {i.id for i in late.split(sourced).control}
    assert research.id in {i.id for i in DOLCI.split(sourced).control}
