"""Stage 4: assemble the verified-clean MATH-500 baseline.

For each MATH-500 item, computes the maximum similarity to any training item
across all three audited projects, removes anything that appears in C_lex or
C_sem, and selects the N items with the lowest max similarity as the clean
baseline.

Reads:  results/ngram/*_hits.jsonl, results/judge/*_results.jsonl, results/embeddings/*_candidates.jsonl, data/raw/math500.jsonl
Writes: data/processed/clean_baseline.jsonl
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict

import numpy as np

import _common  # noqa: F401

from contamination_audit.config import configure_logging
from contamination_audit.io import REPO_ROOT, load_jsonl, paths, save_jsonl

_log = logging.getLogger("build_clean_set")


def load_contaminated_ids(projects: list[str]) -> set[str]:
    contaminated: set[str] = set()
    for project in projects:
        p = paths(project)
        if p.ngram_hits.exists():
            for item in load_jsonl(p.ngram_hits):
                contaminated.add(item["math500_id"])
        if p.judge_results.exists():
            for item in load_jsonl(p.judge_results):
                if item.get("classification") == "CONTAMINATED":
                    contaminated.add(item["math500_id"])
    return contaminated


def max_similarities(projects: list[str]) -> dict[str, float]:
    sims: dict[str, float] = defaultdict(float)
    for project in projects:
        p = paths(project)
        if not p.embedding_candidates.exists():
            continue
        for item in load_jsonl(p.embedding_candidates):
            mid = item["math500_id"]
            sims[mid] = max(sims[mid], item["similarity_score"])
    return dict(sims)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=100,
                        help="Number of clean items to select (default: 100)")
    parser.add_argument("--projects", nargs="+", default=["s1", "tulu", "openthoughts"])
    args = parser.parse_args()

    configure_logging()

    contaminated = load_contaminated_ids(args.projects)
    sims = max_similarities(args.projects)
    _log.info("contaminated items across all projects: %d", len(contaminated))

    math500 = load_jsonl(paths("s1").math500)
    clean_pool = [
        {**item, "max_similarity": sims.get(item["math500_id"], 0.0)}
        for item in math500
        if item["math500_id"] not in contaminated
    ]
    _log.info("clean pool: %d items", len(clean_pool))

    clean_pool.sort(key=lambda x: x["max_similarity"])
    selected = clean_pool[: args.target]
    if not selected:
        _log.error("no clean candidates found — embedding stage must run first")
        return

    sims_arr = [item["max_similarity"] for item in selected]
    _log.info("selected %d clean items (max sim %.3f, mean sim %.3f)",
              len(selected), max(sims_arr), float(np.mean(sims_arr)))

    counts = Counter(item["subject"] for item in selected)
    for subj, count in counts.most_common():
        _log.info("  %-25s %d", subj, count)

    out = REPO_ROOT / "data" / "processed" / "clean_baseline.jsonl"
    save_jsonl(selected, out)
    _log.info("saved -> %s", out)


if __name__ == "__main__":
    main()
