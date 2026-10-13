"""Verify and summarize the saved judge responses without API calls or dependencies."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows, input_path):
    summary = {
        "scope": (
            f"Model judgments of {len(rows)} saved candidate pairs, with up to three saved candidates per retrieval branch. "
            "Labels are not human ground truth; questions without a confirmed saved match are not established negatives."
        ),
        "temporal_controls": (
            "Control benchmark items remain negatives under the study temporal assumption. "
            "Semantic relationship labels do not change their temporal labels."
        ),
        "pairs": len(rows),
        "labels": dict(Counter(r["judgment"]["label"] for r in rows)),
        "needs_review": sum(r["needs_review"] for r in rows),
        "quote_check_failed_or_missing": sum(not r["quotes_match"] for r in rows),
        "groups": {}, "retrieval_branches": {},
        "token_usage": usage(rows),
        "model": sorted({r["model"] for r in rows}),
        "prompt_sha256": hashlib.sha256(rows[0]["prompt"].encode()).hexdigest(),
        "input_sha256": sha256(input_path),
    }
    for group in ("measure", "control"):
        selected = [r for r in rows if any(b["group"] == group for b in r["benchmark_items"])]
        questions = defaultdict(list)
        for row in selected:
            questions[row["benchmark_problem"]].append(row["judgment"]["label"])
        summary["groups"][group] = {
            "pairs": len(selected),
            "labels": dict(Counter(r["judgment"]["label"] for r in selected)),
            "overlap_types": dict(Counter(r["judgment"]["overlap_type"] for r in selected)),
            "distinct_questions": len(questions),
            "with_at_least_one_same_problem_pair": sum("same_problem" in labels for labels in questions.values()),
            "with_at_least_one_uncertain_pair": sum("uncertain" in labels for labels in questions.values()),
        }
    for branch in sorted({m for r in rows for m in r["retrieved_by"]}):
        selected = [r for r in rows if branch in r["retrieved_by"]]
        summary["retrieval_branches"][branch] = {
            "pairs": len(selected),
            "labels": dict(Counter(r["judgment"]["label"] for r in selected)),
            "scope": (
                "Saved pairs retrieved by this branch; branches overlap. "
                "These are not benchmark-level precision estimates."
            ),
        }
    return summary


def usage(rows):
    return {key: sum(r["usage"][key] for r in rows) for key in ("input_tokens", "output_tokens", "total_tokens")}


def summarize_review(primary, reviews, sample):
    first = {r["pair_id"]: r for r in primary}
    second = {r["pair_id"]: r for r in reviews}
    random_ids = set(sample["pair_ids"])
    required = random_ids | {r["pair_id"] for r in primary if r["needs_review"]}
    assert len(second) == len(reviews) and set(second) == required

    def compare(ids):
        pairs = [(first[i]["judgment"], second[i]["judgment"]) for i in ids]
        return {
            "pairs": len(ids),
            "relationship_label_agreement": sum(a["label"] == b["label"] for a, b in pairs),
            "same_problem_binary_agreement": sum(
                (a["label"] == "same_problem") == (b["label"] == "same_problem") for a, b in pairs
            ),
            "overlap_type_agreement": sum(a["overlap_type"] == b["overlap_type"] for a, b in pairs),
        }

    return {
        "scope": (
            "Independent same-model repeat with identical prompt; neither first-pass labels nor filter metadata were shown. "
            "Agreement measures consistency, not accuracy or ground truth. Primary labels are preserved."
        ),
        "random_sample": compare(random_ids),
        "targeted_cases": compare(required - random_ids),
        "all_reviewed": compare(required),
        "disagreement_pair_ids": sorted(i for i in required if any(
            first[i]["judgment"][key] != second[i]["judgment"][key] for key in ("label", "overlap_type")
        )),
        "review_pass_needs_review": [r["pair_id"] for r in reviews if r["needs_review"]],
        "token_usage": usage(reviews),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results/llm_judge")
    parser.add_argument("--output-dir", type=Path, help="Write rebuilt summaries to a separate directory")
    args = parser.parse_args()
    folder = args.data
    inputs = read_rows(folder / "pairs.jsonl")
    rows = read_rows(folder / "labels.jsonl")
    reviews = read_rows(folder / "review_labels.jsonl")
    sample = json.loads((folder / "review_random_sample.json").read_text())
    manifest = json.loads((folder / "run_manifest.json").read_text())
    source_manifest = json.loads((folder / "input_manifest.json").read_text())
    by_id = {r["pair_id"]: r for r in inputs}
    assert len(by_id) == len(inputs)
    assert [r["pair_id"] for r in rows] == list(by_id)
    assert sha256(folder / "pairs.jsonl") == manifest["input_sha256"] == source_manifest["pairs_sha256"]
    assert sha256(folder / "labels.jsonl") == manifest["output_sha256"]
    assert sha256(folder / source_manifest["source"]) == source_manifest["source_sha256"]
    assert sha256(ROOT / "scripts/06_llm_judge.py") == manifest["script_sha256"]
    assert random.Random(sample["seed"]).sample(list(by_id), len(sample["pair_ids"])) == sample["pair_ids"]
    review_ids = set(sample["pair_ids"]) | {r["pair_id"] for r in rows if r["needs_review"]}
    assert read_rows(folder / "review_pairs.jsonl") == [r for r in inputs if r["pair_id"] in review_ids]
    for row in rows + reviews:
        assert all(row[key] == value for key, value in by_id[row["pair_id"]].items())
        assert hashlib.sha256(row["prompt"].encode()).hexdigest() == manifest["prompt_sha256"]
        judgment = row["judgment"]
        quotes_match = bool(judgment["quote_a"] and judgment["quote_b"]) and (
            judgment["quote_a"] in row["benchmark_problem"] and judgment["quote_b"] in row["training_problem"]
        )
        assert quotes_match == row["quotes_match"]
    outputs = {
        "summary.json": summarize(rows, folder / "pairs.jsonl"),
        "review_summary.json": summarize_review(rows, reviews, sample),
    }
    for name, result in outputs.items():
        assert result == json.loads((folder / name).read_text()), f"Saved {name} differs"
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with (args.output_dir / name).open("x", encoding="utf-8") as output:
                output.write(json.dumps(result, indent=2) + "\n")
    print(f"Verified {len(rows)} judgments and {len(reviews)} repeats; both summaries match.")
    print(json.dumps(outputs["summary.json"]["labels"]))
    print(json.dumps(outputs["review_summary.json"]["random_sample"]))


if __name__ == "__main__":
    main()
