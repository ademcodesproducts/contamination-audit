"""Stage 5: assemble final C_lex / C_sem sets and emit Tables 1 + 2 for the paper.

This is the canonical reproducer for the paper's contamination counts.

C_lex is deduped by math500_id with the strongest hit retained; ``--min-ngrams``
controls the lower bound on shared n-grams (default 5, the conservative choice
written to file). The strict (≥10) and raw (≥1) variants are reported alongside
for sensitivity analysis.

Optionally runs a "crosscheck" diagnostic (folded in from the deleted
06_crosscheck_ngram.py) that classifies each judge-CONTAMINATED pair as
caught-by-both (filter bug) or semantic-only (paper's core finding).

Reads:  results/ngram/*_hits.jsonl, results/judge/*_results.jsonl
Writes: data/processed/{project}_c_lex.jsonl, data/processed/{project}_c_sem.jsonl,
        results/tables/table2_contamination_counts.csv,
        results/tables/crosscheck.csv (if --crosscheck)
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

import _common  # noqa: F401

from contamination_audit.config import configure_logging
from contamination_audit.io import REPO_ROOT, load_jsonl, paths, save_jsonl
from contamination_audit.ngram import dedup_by_math500

_log = logging.getLogger("validate_and_report")


DATASET_SIZES = {
    "s1": 1_000,
    "tulu": 84_312,
    "openthoughts": 113_957,
    "openthoughts_full": 113_957,
}


def _pick_project(project: str) -> str:
    """Prefer the *_full ngram/judge files when both are present (OpenThoughts)."""
    if project != "openthoughts":
        return project
    full = paths("openthoughts_full")
    return "openthoughts_full" if full.ngram_hits.exists() and full.judge_results.exists() else project


def assemble(min_ngrams: int) -> pd.DataFrame:
    rows = []
    for raw_project in ["s1", "tulu", "openthoughts"]:
        project = _pick_project(raw_project)
        p = paths(project)
        train_size = DATASET_SIZES[project]

        # ── C_lex ─────────────────────────────────────────────────────────────
        if p.ngram_hits.exists():
            raw = load_jsonl(p.ngram_hits)
            c_lex = dedup_by_math500(raw, min_ngrams)
            c_lex_strict = dedup_by_math500(raw, 10)
            c_lex_loose = dedup_by_math500(raw, 1)
            out_lex = REPO_ROOT / "data" / "processed" / f"{raw_project}_c_lex.jsonl"
            save_jsonl(c_lex, out_lex)
            ngram_n = c_lex[0]["ngram_n"] if c_lex else (raw[0]["ngram_n"] if raw else "?")
            rows.append({
                "project": raw_project,
                "contamination_type": "C_lex",
                "n_unique_math500": len(c_lex),
                "n_lex_strict_10": len(c_lex_strict),
                "n_lex_raw_1": len(c_lex_loose),
                "train_size": train_size,
                "rate_pct": round(len(c_lex) / 500 * 100, 2),
                "judge_precision": "N/A (lexical)",
                "threshold_ngrams": min_ngrams,
                "notes": f"n={ngram_n}",
            })
            _log.info("%s C_lex: %d (>=%d) | %d (>=10) | %d (>=1)",
                      raw_project, len(c_lex), min_ngrams, len(c_lex_strict), len(c_lex_loose))
        else:
            _log.warning("no n-gram hits for %s", raw_project)

        # ── C_sem ─────────────────────────────────────────────────────────────
        if p.judge_results.exists():
            judged = load_jsonl(p.judge_results)
            c_sem_all = [j for j in judged if j.get("classification") == "CONTAMINATED"]
            c_sem_all.sort(key=lambda x: -x.get("similarity_score", 0))
            seen: set[str] = set()
            c_sem = []
            for item in c_sem_all:
                if item["math500_id"] not in seen:
                    seen.add(item["math500_id"])
                    c_sem.append(item)
            out_sem = REPO_ROOT / "data" / "processed" / f"{raw_project}_c_sem.jsonl"
            save_jsonl(c_sem, out_sem)

            total = sum(1 for j in judged if j.get("classification") != "ERROR")
            precision = len(c_sem_all) / max(total, 1)
            rows.append({
                "project": raw_project,
                "contamination_type": "C_sem",
                "n_unique_math500": len(c_sem),
                "n_lex_strict_10": "",
                "n_lex_raw_1": "",
                "train_size": train_size,
                "rate_pct": round(len(c_sem) / 500 * 100, 2),
                "judge_precision": f"{precision:.0%}",
                "threshold_ngrams": "",
                "notes": f"{total} pairs judged",
            })
            _log.info("%s C_sem: %d unique (precision %.0f%%)", raw_project, len(c_sem), precision * 100)
        else:
            _log.warning("no judge results for %s", raw_project)

    df = pd.DataFrame(rows)
    out = REPO_ROOT / "results" / "tables" / "table2_contamination_counts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    _log.info("Table 2 -> %s", out)
    print()
    print(df.to_string(index=False))
    return df


def crosscheck() -> None:
    """Cross-check judge-CONTAMINATED pairs against n-gram hits per project.

    Carried over from the deleted top-level 06_crosscheck_ngram.py: tells us
    which judge-flagged pairs the n-gram filter would also have caught (failure
    mode: implementation bug) versus which the n-gram filter missed (failure
    mode: paraphrase / template — the paper's headline finding).
    """
    rows = []
    for raw_project in ["s1", "tulu", "openthoughts"]:
        project = _pick_project(raw_project)
        p = paths(project)
        if not (p.ngram_hits.exists() and p.judge_results.exists()):
            continue

        ngram_pairs = {(item["math500_id"], item["train_id"]) for item in load_jsonl(p.ngram_hits)}
        for item in load_jsonl(p.judge_results):
            if item.get("classification") != "CONTAMINATED":
                continue
            pair = (item["math500_id"], item["train_id"])
            rows.append({
                "project": raw_project,
                "math500_id": item["math500_id"],
                "train_id": item["train_id"],
                "similarity_score": round(item.get("similarity_score", 0), 3),
                "subject": item.get("math500_subject", ""),
                "caught_by_ngram": pair in ngram_pairs,
                "failure_mode": "implementation_bug" if pair in ngram_pairs else "paraphrase_or_template",
                "reasoning": item.get("reasoning", ""),
            })

    df = pd.DataFrame(rows)
    out = REPO_ROOT / "results" / "tables" / "crosscheck.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if not df.empty:
        bug = (df["failure_mode"] == "implementation_bug").sum()
        sem = (df["failure_mode"] == "paraphrase_or_template").sum()
        _log.info("crosscheck: %d implementation_bug | %d paraphrase_or_template", bug, sem)
    _log.info("crosscheck -> %s", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-ngrams", type=int, default=5,
                        help="Min shared n-grams for C_lex (default: 5, the paper's conservative choice)")
    parser.add_argument("--crosscheck", action="store_true",
                        help="Also emit results/tables/crosscheck.csv")
    args = parser.parse_args()

    configure_logging()
    assemble(args.min_ngrams)
    if args.crosscheck:
        crosscheck()


if __name__ == "__main__":
    main()
