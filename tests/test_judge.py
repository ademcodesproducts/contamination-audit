"""The frozen LLM-judge record, and the tooling that rebuilds and checks it."""

import json
import shutil

import pytest

from contamination_audit.errors import IntegrityError
from contamination_audit.judge import CandidatePairBuilder, JudgeRecord

from .conftest import REPO

RECORD = REPO / "results" / "llm_judge"


@pytest.fixture
def record_copy(tmp_path):
    return shutil.copytree(RECORD, tmp_path / "llm_judge")


def test_the_published_record_verifies():
    JudgeRecord(RECORD).verify()


def test_pairs_rebuild_byte_for_byte_from_the_retrieval_snapshot():
    matches = json.loads((RECORD / "candidate_matches.json").read_text(encoding="utf-8"))
    rebuilt = "".join(
        json.dumps(pair, ensure_ascii=False) + "\n"
        for pair in CandidatePairBuilder(matches).build()
    )
    assert rebuilt.encode() == (RECORD / "pairs.jsonl").read_bytes()


def test_an_edited_label_is_caught(record_copy):
    labels = record_copy / "labels.jsonl"
    labels.write_bytes(labels.read_bytes().replace(b"generic_overlap", b"same_problem", 1))
    with pytest.raises(IntegrityError, match="labels.jsonl"):
        JudgeRecord(record_copy).verify()


def test_a_line_ending_rewrite_is_caught(record_copy):
    pairs = record_copy / "pairs.jsonl"
    pairs.write_bytes(pairs.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(IntegrityError, match="line endings"):
        JudgeRecord(record_copy).verify()


def match(group, index, problem, hits, flagged=True):
    """One item of a retrieval snapshot, in candidate_matches.json's shape."""
    method = {"flag": flagged, "top": hits}
    return {
        "group": group,
        "index": index,
        "benchmark": {"id": f"{group}{index}", "problem": problem},
        "methods": {
            "s1": method,
            "ot3": method,
            "coverage": {"flag": False, "top": [], "union_coverage": 0},
        },
        "fuzzy_top": [],
    }


def test_identical_text_pairs_are_merged_keeping_every_membership():
    hit = {"training": {"train_id": "t1", "problem": "same training text"}, "ngram": "x"}
    matches = [
        match("measure", 0, "duplicated benchmark text", [hit]),
        match("measure", 7, "duplicated benchmark text", [hit]),
    ]
    (pair,) = CandidatePairBuilder(matches).build()
    assert [item["index"] for item in pair["benchmark_items"]] == [0, 7]
    assert pair["training_ids"] == ["t1"]
    assert pair["retrieved_by"] == ["s1", "ot3"]


def test_unflagged_items_contribute_no_pairs():
    hit = {"training": {"train_id": "t1", "problem": "text"}, "ngram": "x"}
    assert CandidatePairBuilder([match("control", 0, "q", [hit], flagged=False)]).build() == []
