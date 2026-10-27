"""Rebuild the judge's input pairs from the frozen retrieval snapshot.

`candidate_matches.json` holds, for each of the 782 benchmark items, the top
three training documents each filter branch retrieved. This selects the pairs
worth judging, merges identical (benchmark, training) text pairs while keeping
every benchmark membership and retrieval branch, and attaches all the saved
evidence for each pair. No filter is run and no API is called.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

COVERAGE_CUTOFF = 0.5  # Tulu 3's "more than half", per document
FUZZY_CUTOFF = 75  # OpenThoughts3's fuzzy branch

Row = dict[str, Any]


def pair_id(benchmark_problem: str, training_problem: str) -> str:
    key = json.dumps((benchmark_problem, training_problem), ensure_ascii=False)
    return hashlib.sha256(key.encode()).hexdigest()[:20]


class CandidatePairBuilder:
    def __init__(self, matches: list[Row]) -> None:
        self.matches = matches
        self.pairs: dict[tuple[str, str], Row] = {}

    def build(self) -> list[Row]:
        for row in self.matches:
            self._add_candidates(row)
        # Evidence comes second: a branch's saved match is attached to a pair
        # even when that branch was not what selected the pair.
        for row in self.matches:
            self._attach_evidence(row)
        return list(self.pairs.values())

    # ------------------------------------------------------- selection

    @staticmethod
    def _candidates(row: Row) -> list[tuple[str, Row]]:
        """(branch, saved hit) for every candidate this benchmark item contributes."""
        methods = row["methods"]
        candidates = []
        for branch in ("s1", "ot3"):
            if methods[branch]["flag"]:
                candidates += [(branch, hit) for hit in methods[branch]["top"]]
        candidates += [
            ("coverage_per_document", hit)
            for hit in methods["coverage"]["top"]
            if hit["coverage"] > COVERAGE_CUTOFF
        ]
        candidates += [
            ("ot3_fuzzy75", hit) for hit in row["fuzzy_top"] if hit["score"] >= FUZZY_CUTOFF
        ]
        return candidates

    def _add_candidates(self, row: Row) -> None:
        benchmark_problem = row["benchmark"]["problem"]
        for branch, hit in self._candidates(row):
            training = hit["training"]
            key = (benchmark_problem, training["problem"])
            if key not in self.pairs:
                self.pairs[key] = {
                    "pair_id": pair_id(*key),
                    "benchmark_problem": benchmark_problem,
                    "training_problem": training["problem"],
                    "benchmark_items": [],
                    "training_ids": [],
                    "retrieved_by": [],
                    "filter_evidence": [],
                }
            pair = self.pairs[key]
            item = {"group": row["group"], "index": row["index"], "id": row["benchmark"]["id"]}
            for field, value in (
                ("benchmark_items", item),
                ("training_ids", training["train_id"]),
                ("retrieved_by", branch),
            ):
                if value not in pair[field]:
                    pair[field].append(value)

    # -------------------------------------------------------- evidence

    @staticmethod
    def _item_flags(row: Row) -> tuple[dict, dict, dict]:
        """The item's verdicts under the original audit and the corrected filters."""
        methods = row["methods"]
        best_fuzzy = max((hit["score"] for hit in row["fuzzy_top"]), default=0)
        best_coverage = max((hit["coverage"] for hit in methods["coverage"]["top"]), default=0)
        original = {
            "s1-8gram-word": methods["s1"]["flag"],
            "ot3-13gram-qwen": methods["ot3"]["flag"],
            "ot114k-fuzz95": best_fuzzy >= 95,
            "tulu3-8gram-50pct": methods["coverage"]["flag"],
            "decon-default": False,
        }
        corrected = {
            "s1-8gram-word": original["s1-8gram-word"],
            "ot114k-fuzz95": original["ot114k-fuzz95"],
            "decon-default": False,
            "ot3-paper-or75": original["ot3-13gram-qwen"] or best_fuzzy >= FUZZY_CUTOFF,
            "coverage-best-document-adaptation": best_coverage > COVERAGE_CUTOFF,
        }
        scores = {
            "max_fuzzy_ratio": best_fuzzy,
            "coverage_union": methods["coverage"]["union_coverage"],
            "coverage_best_document": best_coverage,
        }
        return original, corrected, scores

    def _attach_evidence(self, row: Row) -> None:
        original, corrected, scores = self._item_flags(row)
        benchmark_problem = row["benchmark"]["problem"]
        branches = [(name, data["top"]) for name, data in row["methods"].items()]
        branches.append(("fuzzy", row["fuzzy_top"]))

        saved_matches: dict[tuple[str, str], list[Row]] = {}
        for method, hits in branches:
            for hit in hits:
                key = (benchmark_problem, hit["training"]["problem"])
                if key in self.pairs:
                    saved_matches.setdefault(key, []).append(
                        {
                            "method": method,
                            "train_id": hit["training"]["train_id"],
                            **{k: v for k, v in hit.items() if k != "training"},
                        }
                    )

        for key, matches in saved_matches.items():
            self.pairs[key]["filter_evidence"].append(
                {
                    "group": row["group"],
                    "benchmark_index": row["index"],
                    "original_item_flags": original,
                    "corrected_item_flags": corrected,
                    "item_scores": scores,
                    "saved_pair_matches": matches,
                }
            )
