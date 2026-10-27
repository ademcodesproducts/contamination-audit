"""Rebuild the judge inputs from the frozen retrieval snapshot; no filter or API calls."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_pairs(matches):
    pairs = {}
    for row in matches:
        problem = row["benchmark"]["problem"]
        methods = row["methods"]
        candidates = []
        for branch in ("s1", "ot3"):
            if methods[branch]["flag"]:
                candidates.extend((branch, hit) for hit in methods[branch]["top"])
        candidates.extend(
            ("coverage_per_document", hit)
            for hit in methods["coverage"]["top"] if hit["coverage"] > 0.5
        )
        candidates.extend(
            ("ot3_fuzzy75", hit) for hit in row["fuzzy_top"] if hit["score"] >= 75
        )
        for branch, hit in candidates:
            training = hit["training"]
            key = (problem, training["problem"])
            if key not in pairs:
                pairs[key] = {
                    "pair_id": hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:20],
                    "benchmark_problem": problem,
                    "training_problem": training["problem"],
                    "benchmark_items": [], "training_ids": [], "retrieved_by": [],
                    "filter_evidence": [],
                }
            pair = pairs[key]
            item = {"group": row["group"], "index": row["index"], "id": row["benchmark"]["id"]}
            memberships = [
                ("benchmark_items", item),
                ("training_ids", training["train_id"]),
                ("retrieved_by", branch),
            ]
            for field, value in memberships:
                if value not in pair[field]:
                    pair[field].append(value)

    # Keep all saved evidence for each selected text pair, including other branches.
    for row in matches:
        methods = row["methods"]
        fuzzy = max((hit["score"] for hit in row["fuzzy_top"]), default=0)
        coverage = max((hit["coverage"] for hit in methods["coverage"]["top"]), default=0)
        original = {
            "s1-8gram-word": methods["s1"]["flag"],
            "ot3-13gram-qwen": methods["ot3"]["flag"],
            "ot114k-fuzz95": fuzzy >= 95,
            "tulu3-8gram-50pct": methods["coverage"]["flag"],
            "decon-default": False,
        }
        corrected = {
            "s1-8gram-word": original["s1-8gram-word"],
            "ot114k-fuzz95": original["ot114k-fuzz95"],
            "decon-default": False,
            "ot3-paper-or75": original["ot3-13gram-qwen"] or fuzzy >= 75,
            "coverage-best-document-adaptation": coverage > 0.5,
        }
        evidence = {}
        branches = [(name, data["top"]) for name, data in methods.items()]
        branches.append(("fuzzy", row["fuzzy_top"]))
        for method, hits in branches:
            for hit in hits:
                key = (row["benchmark"]["problem"], hit["training"]["problem"])
                if key in pairs:
                    evidence.setdefault(key, []).append({
                        "method": method, "train_id": hit["training"]["train_id"],
                        **{k: v for k, v in hit.items() if k != "training"},
                    })
        for key, hits in evidence.items():
            pairs[key]["filter_evidence"].append({
                "group": row["group"], "benchmark_index": row["index"],
                "original_item_flags": original, "corrected_item_flags": corrected,
                "item_scores": {
                    "max_fuzzy_ratio": fuzzy,
                    "coverage_union": methods["coverage"]["union_coverage"],
                    "coverage_best_document": coverage,
                },
                "saved_pair_matches": hits,
            })
    return list(pairs.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/llm_judge/candidate_matches.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decon = json.loads((ROOT / "results/paper/decon.json").read_text(encoding="utf-8"))
    assert decon["measure"] == decon["control"] == 0, "This snapshot requires the saved zero-result decon run"
    pairs = build_pairs(json.loads(args.source.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for pair in pairs:
            output.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
