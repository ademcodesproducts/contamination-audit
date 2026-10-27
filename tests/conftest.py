"""A miniature study on disk, small enough that every answer is known in advance.

Dolci measurement items, in order:
    0  olymmath        reproduced verbatim by a training document
    1  olymmath
    2  aime_2024_I
    3  aime_2025       shares the AIME stock phrase with a training document
    4  hmmt_feb_2025
Dolci control items:
    0  aime_2026       shares the same stock phrase, so it is a false positive
    1  hmmt_feb_2026
    2  arxivmath-0126
"""

import json
from pathlib import Path

import pytest

from contamination_audit.experiments import Study
from contamination_audit.filters import (
    S1_WORDS,
    FuzzyRatio,
    Provenance,
    PublishedFilter,
    SharedNgram,
)
from contamination_audit.paths import ProjectPaths

REPO = Path(__file__).resolve().parents[1]

STOCK_PHRASE = "let m and n be relatively prime positive integers find m plus n"
VERBATIM = "olympiad problem alpha beta gamma delta epsilon zeta eta theta"


def item(benchmark: str, index: int, problem: str) -> dict:
    return {
        "id": f"{benchmark}_{index:04d}",
        "benchmark": benchmark,
        "problem": problem,
        "answer": "0",
    }


BENCHMARK_FILES = {
    "olymmath_en_hard.jsonl": [{"id": "olym_en_hard_0000", "problem": VERBATIM, "answer": "1"}],
    "olymmath_en_easy.jsonl": [
        {
            "id": "olym_en_easy_0000",
            "problem": "count the ways to tile a two by ten board",
            "answer": "2",
        }
    ],
    "matharena_measurement.jsonl": [
        item("aime_2024_I", 0, "find the remainder when the number is divided by one thousand"),
        item("aime_2025", 0, f"for this new configuration {STOCK_PHRASE}"),
        item("hmmt_feb_2025", 0, "a triangle has sides three four and five so find its area"),
    ],
    "matharena_control.jsonl": [
        item("aime_2026", 0, f"compute the probability and {STOCK_PHRASE}"),
        item("hmmt_feb_2026", 0, "a fresh question sharing nothing with any training text"),
        item("arxivmath-0126", 0, "show that every compact operator on a hilbert space is bounded"),
    ],
    "matharena_control_post_may2026.jsonl": [
        item("arxivmath-0826", 0, "a research question published long after every corpus"),
    ],
}

CORPUS = [
    VERBATIM,
    f"prove that if {STOCK_PHRASE} then the claim holds",
    "an unrelated document about cooking pasta",
    "",
]

CHEAP_FILTERS = [
    PublishedFilter("words-8gram", "test", Provenance.CODE, SharedNgram(8, S1_WORDS), "8-gram"),
    PublishedFilter("fuzzy-95", "test", Provenance.PROSE, FuzzyRatio(95), "whole string"),
]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


@pytest.fixture
def tiny_root(tmp_path: Path) -> Path:
    for filename, records in BENCHMARK_FILES.items():
        write_jsonl(tmp_path / "data" / "benchmarks" / filename, records)
    documents = [
        {"train_id": f"doc{i}", "problem": text, "source": "a" if i % 2 else "b"}
        for i, text in enumerate(CORPUS)
    ]
    write_jsonl(tmp_path / "data" / "corpora" / "dolci.jsonl", documents)
    decon = {"corpus_rows": len(CORPUS), "n_measure": 5, "n_control": 3, "measure": 0, "control": 0}
    (tmp_path / "results" / "paper").mkdir(parents=True)
    (tmp_path / "results" / "paper" / "decon.json").write_text(json.dumps(decon), encoding="utf-8")
    return tmp_path


@pytest.fixture
def tiny_study(tiny_root: Path) -> Study:
    """A study whose filters are cheap: no model tokenizers to download."""
    return Study(ProjectPaths.at(tiny_root), filters=CHEAP_FILTERS, corpus_size=len(CORPUS))
