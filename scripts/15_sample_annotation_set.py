"""Stage 15: draw a blind, stratified annotation sample from judge candidate pairs.

Emits one worksheet per annotator with the judge verdict and similarity score
withheld -- showing either would anchor the annotator against the very signal the
annotation exists to evaluate. The mapping back to judge verdicts is written
separately as a key file and must not be opened before both passes are complete.

Reads:  results/judge/*_judge_results.jsonl
Writes: results/annotations/worksheet_{annotator}.csv
        results/annotations/sample_key.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import _common  # noqa: F401

from contamination_audit.config import configure_logging
from contamination_audit.io import load_jsonl

_log = logging.getLogger("sample_annotation")

JUDGE_DIR = Path("results/judge")
OUT_DIR = Path("results/annotations")

# Bands are on retrieval similarity. Stratifying across them keeps the sample from
# collapsing onto near-duplicates, where every label is trivially CONTAMINATED.
BANDS = [(0.0, 0.75, "low"), (0.75, 0.85, "mid"), (0.85, 0.95, "high"), (0.95, 1.01, "near_dup")]

WORKSHEET_COLS = [
    "pair_id", "benchmark_problem", "benchmark_answer",
    "training_problem", "training_solution_excerpt",
    "label", "confidence", "notes",
]


def band_of(score: float) -> str:
    for lo, hi, name in BANDS:
        if lo <= score < hi:
            return name
    return "unknown"


def load_candidates() -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for path in sorted(JUDGE_DIR.glob("*_judge_results.jsonl")):
        for row in load_jsonl(path):
            key = (row.get("project"), row.get("math500_id"), row.get("train_id"))
            if key in seen or None in key:
                continue
            seen.add(key)
            row["_source_file"] = path.name
            out.append(row)
    return out


def stratified_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample evenly across project x band, redistributing any shortfall."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["project"], band_of(float(r.get("similarity_score") or 0.0)))].append(r)

    picked: list[dict] = []
    quota = max(1, n // max(1, len(buckets)))
    for key in sorted(buckets):
        pool = buckets[key]
        take = min(quota, len(pool))
        picked.extend(rng.sample(pool, take))

    # Redistribute whatever the per-bucket cap left on the table.
    if len(picked) < n:
        chosen = {id(r) for r in picked}
        rest = [r for r in rows if id(r) not in chosen]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--annotators", nargs="+", default=["a1", "a2"])
    args = ap.parse_args()
    configure_logging()

    rng = random.Random(args.seed)
    rows = load_candidates()
    _log.info("loaded %d unique candidate pairs", len(rows))

    by_proj = Counter(r["project"] for r in rows)
    for proj, cnt in sorted(by_proj.items()):
        _log.info("  %-20s %4d", proj, cnt)

    sample = stratified_sample(rows, args.n, rng)
    for i, r in enumerate(sample):
        r["pair_id"] = f"pair_{i:04d}"
    rng.shuffle(sample)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for ann_i, ann in enumerate(args.annotators):
        # Each annotator sees a different order. A shared order would make any
        # drift over the session correlated between annotators, inflating kappa.
        ordered = list(sample)
        random.Random(args.seed + 1000 + ann_i).shuffle(ordered)
        path = OUT_DIR / f"worksheet_{ann}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=WORKSHEET_COLS)
            w.writeheader()
            for r in ordered:
                w.writerow({
                    "pair_id": r["pair_id"],
                    "benchmark_problem": r.get("math500_problem", ""),
                    "benchmark_answer": r.get("math500_answer", ""),
                    "training_problem": r.get("train_problem", ""),
                    "training_solution_excerpt": (r.get("train_solution") or "")[:1500],
                    "label": "", "confidence": "", "notes": "",
                })
        _log.info("wrote %s (%d pairs)", path, len(sample))

    key_path = OUT_DIR / "sample_key.csv"
    with key_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "project", "math500_id", "train_id",
                    "similarity_score", "band", "judge_classification",
                    "judge_confidence", "source_file"])
        for r in sample:
            score = float(r.get("similarity_score") or 0.0)
            w.writerow([r["pair_id"], r["project"], r["math500_id"], r["train_id"],
                        f"{score:.4f}", band_of(score), r.get("classification", ""),
                        r.get("confidence", ""), r["_source_file"]])
    _log.info("wrote %s  (DO NOT OPEN until both annotation passes are done)", key_path)

    dist = Counter((r["project"], band_of(float(r.get("similarity_score") or 0.0))) for r in sample)
    _log.info("sample composition:")
    for k in sorted(dist):
        _log.info("  %-22s %-10s %3d", k[0], k[1], dist[k])


if __name__ == "__main__":
    main()
