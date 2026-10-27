"""The frozen LLM-judge experiment, and the checks that it is intact.

results/llm_judge/ holds a completed run: 565 benchmark/training pairs, the
model's judgment of each, a same-model repeat of 20 pairs, and manifests that
record SHA-256 hashes of the inputs, the outputs, the prompt, and the script
that made the calls. `JudgeRecord.verify` recomputes every one of them, so the
published labels can be trusted to be the ones that run produced.

The judge script is kept byte for byte as it ran, in run_judge.py beside its
outputs. Editing it, even to fix style, would break the hash that ties it to
the labels.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from ..errors import IntegrityError
from ..jsonio import read_json, read_jsonl

JUDGE_SCRIPT = "run_judge.py"
SUMMARIES = ("summary.json", "review_summary.json")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quotes_are_verbatim(row: dict[str, Any]) -> bool:
    """Whether the judge's supporting quotes appear in the texts it was shown."""
    judgment = row["judgment"]
    return bool(judgment["quote_a"] and judgment["quote_b"]) and (
        judgment["quote_a"] in row["benchmark_problem"]
        and judgment["quote_b"] in row["training_problem"]
    )


def token_usage(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = ("input_tokens", "output_tokens", "total_tokens")
    return {key: sum(row["usage"][key] for row in rows) for key in keys}


@dataclass
class JudgeRecord:
    folder: Path

    @cached_property
    def pairs(self) -> list[dict]:
        return read_jsonl(self.folder / "pairs.jsonl")

    @cached_property
    def labels(self) -> list[dict]:
        return read_jsonl(self.folder / "labels.jsonl")

    @cached_property
    def review_labels(self) -> list[dict]:
        return read_jsonl(self.folder / "review_labels.jsonl")

    @cached_property
    def review_sample(self) -> dict:
        return read_json(self.folder / "review_random_sample.json")

    @cached_property
    def run_manifest(self) -> dict:
        return read_json(self.folder / "run_manifest.json")

    @cached_property
    def input_manifest(self) -> dict:
        return read_json(self.folder / "input_manifest.json")

    # ---------------------------------------------------------- checks

    def verify(self) -> None:
        """Raise IntegrityError at the first thing that does not match its record."""
        run, inputs = self.run_manifest, self.input_manifest
        by_id = {pair["pair_id"]: pair for pair in self.pairs}

        self._expect(len(by_id) == len(self.pairs), "pair ids are not unique")
        self._expect(
            [row["pair_id"] for row in self.labels] == list(by_id),
            "labels.jsonl is not aligned with pairs.jsonl",
        )
        self._expect_hash("pairs.jsonl", run["input_sha256"])
        self._expect_hash("pairs.jsonl", inputs["pairs_sha256"])
        self._expect_hash("labels.jsonl", run["output_sha256"])
        self._expect_hash(inputs["source"], inputs["source_sha256"])
        self._expect_hash(JUDGE_SCRIPT, run["script_sha256"])

        sample = self.review_sample
        redrawn = random.Random(sample["seed"]).sample(list(by_id), len(sample["pair_ids"]))
        self._expect(
            redrawn == sample["pair_ids"], "the repeat sample does not redraw from its seed"
        )
        review_ids = set(sample["pair_ids"]) | {
            r["pair_id"] for r in self.labels if r["needs_review"]
        }
        self._expect(
            read_jsonl(self.folder / "review_pairs.jsonl")
            == [pair for pair in self.pairs if pair["pair_id"] in review_ids],
            "review_pairs.jsonl is not the sampled and flagged pairs",
        )

        for row in self.labels + self.review_labels:
            pair_id = row["pair_id"]
            self._expect(
                all(row[key] == value for key, value in by_id[pair_id].items()),
                f"response {pair_id} does not carry its input unchanged",
            )
            self._expect(
                hashlib.sha256(row["prompt"].encode()).hexdigest() == run["prompt_sha256"],
                f"response {pair_id} was made with a different prompt",
            )
            self._expect(
                quotes_are_verbatim(row) == row["quotes_match"],
                f"response {pair_id} misreports whether its quotes are verbatim",
            )

        for name, rebuilt in self.summaries().items():
            self._expect(
                rebuilt == read_json(self.folder / name), f"{name} does not match the labels"
            )

    def _expect(self, condition: bool, problem: str) -> None:
        if not condition:
            raise IntegrityError(f"{self.folder}: {problem}")

    def _expect_hash(self, filename: str, recorded: str) -> None:
        self._expect(
            sha256_of(self.folder / filename) == recorded,
            f"{filename} does not match its recorded SHA-256 (line endings rewritten by git?)",
        )

    # ------------------------------------------------------- summaries

    def summaries(self) -> dict[str, dict]:
        return {"summary.json": self.summary(), "review_summary.json": self.review_summary()}

    def summary(self) -> dict[str, Any]:
        rows = self.labels
        summary = {
            "scope": (
                f"Model judgments of {len(rows)} saved candidate pairs, with up to three "
                "saved candidates per retrieval branch. Labels are not human ground truth; "
                "questions without a confirmed saved match are not established negatives."
            ),
            "temporal_controls": (
                "Control benchmark items remain negatives under the study temporal assumption. "
                "Semantic relationship labels do not change their temporal labels."
            ),
            "pairs": len(rows),
            "labels": dict(Counter(row["judgment"]["label"] for row in rows)),
            "needs_review": sum(row["needs_review"] for row in rows),
            "quote_check_failed_or_missing": sum(not row["quotes_match"] for row in rows),
            "groups": {},
            "retrieval_branches": {},
            "token_usage": token_usage(rows),
            "model": sorted({row["model"] for row in rows}),
            "prompt_sha256": hashlib.sha256(rows[0]["prompt"].encode()).hexdigest(),
            "input_sha256": sha256_of(self.folder / "pairs.jsonl"),
        }

        for group in ("measure", "control"):
            selected = [
                row
                for row in rows
                if any(item["group"] == group for item in row["benchmark_items"])
            ]
            labels_per_question = defaultdict(list)
            for row in selected:
                labels_per_question[row["benchmark_problem"]].append(row["judgment"]["label"])
            summary["groups"][group] = {
                "pairs": len(selected),
                "labels": dict(Counter(row["judgment"]["label"] for row in selected)),
                "overlap_types": dict(Counter(row["judgment"]["overlap_type"] for row in selected)),
                "distinct_questions": len(labels_per_question),
                "with_at_least_one_same_problem_pair": sum(
                    "same_problem" in labels for labels in labels_per_question.values()
                ),
                "with_at_least_one_uncertain_pair": sum(
                    "uncertain" in labels for labels in labels_per_question.values()
                ),
            }

        for branch in sorted({branch for row in rows for branch in row["retrieved_by"]}):
            selected = [row for row in rows if branch in row["retrieved_by"]]
            summary["retrieval_branches"][branch] = {
                "pairs": len(selected),
                "labels": dict(Counter(row["judgment"]["label"] for row in selected)),
                "scope": (
                    "Saved pairs retrieved by this branch; branches overlap. "
                    "These are not benchmark-level precision estimates."
                ),
            }
        return summary

    def review_summary(self) -> dict[str, Any]:
        first = {row["pair_id"]: row for row in self.labels}
        second = {row["pair_id"]: row for row in self.review_labels}
        random_ids = set(self.review_sample["pair_ids"])
        reviewed = random_ids | {row["pair_id"] for row in self.labels if row["needs_review"]}
        self._expect(
            len(second) == len(self.review_labels) and set(second) == reviewed,
            "review_labels.jsonl does not cover exactly the pairs selected for review",
        )

        def agreement(ids: set[str]) -> dict[str, int]:
            pairs = [(first[i]["judgment"], second[i]["judgment"]) for i in ids]
            return {
                "pairs": len(ids),
                "relationship_label_agreement": sum(a["label"] == b["label"] for a, b in pairs),
                "same_problem_binary_agreement": sum(
                    (a["label"] == "same_problem") == (b["label"] == "same_problem")
                    for a, b in pairs
                ),
                "overlap_type_agreement": sum(
                    a["overlap_type"] == b["overlap_type"] for a, b in pairs
                ),
            }

        return {
            "scope": (
                "Independent same-model repeat with identical prompt; neither first-pass "
                "labels nor filter metadata were shown. Agreement measures consistency, not "
                "accuracy or ground truth. Primary labels are preserved."
            ),
            "random_sample": agreement(random_ids),
            "targeted_cases": agreement(reviewed - random_ids),
            "all_reviewed": agreement(reviewed),
            "disagreement_pair_ids": sorted(
                i
                for i in reviewed
                if any(
                    first[i]["judgment"][key] != second[i]["judgment"][key]
                    for key in ("label", "overlap_type")
                )
            ),
            "review_pass_needs_review": [
                r["pair_id"] for r in self.review_labels if r["needs_review"]
            ],
            "token_usage": token_usage(self.review_labels),
        }
