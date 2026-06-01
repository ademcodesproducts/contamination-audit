"""Stage 12: failure-mode breakdown, Tulu source attribution, n-gram robustness.

Three sub-commands (selectable via ``--check``):

  failure_mode    Cross-dataset C_lex / C_sem comparison with bootstrap 95% CIs.
                  → results/tables/failure_mode_comparison.csv

  tulu_sources    Splits Tulu contamination by training source (NuminaMath, GSM8K, …).
                  → results/tables/tulu_source_breakdown.csv

  ngram_sweep     Re-runs the OpenThoughts n-gram filter at n=15 and n=20 to show
                  C_lex contamination is stable across threshold choices.
                  → results/ngram/openthoughts_full_hits_n{15,20}.jsonl
                  → results/tables/robustness_ngram_sweep.csv
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

import _common  # noqa: F401

from contamination_audit.config import NgramConfig, configure_logging, load_config, seed_everything
from contamination_audit.io import REPO_ROOT, load_jsonl, paths, save_jsonl
from contamination_audit.ngram import dedup_by_math500, find_hits
from contamination_audit.stats import proportion_bootstrap

_log = logging.getLogger("robustness_checks")

MATH500_SIZE = 500


# ── failure_mode ──────────────────────────────────────────────────────────────

DATASET_LABELS = {
    "s1": "8-gram, any overlap",
    "tulu": "8-gram, >50% token coverage",
    "openthoughts": "13-gram, any overlap",
}


def _classify_mode(n_lex: int, n_sem: int) -> str:
    if n_lex > 0 and n_sem > 0:
        ratio = n_lex / max(n_sem, 1)
        if ratio >= 2.0:
            return "C_lex dominant"
        if ratio <= 0.5:
            return "C_sem dominant"
        return "mixed"
    if n_lex > 0:
        return "C_lex only"
    return "C_sem only"


def failure_mode(*, n_bootstrap: int, seed: int) -> None:
    rows = []
    for project in ["s1", "tulu", "openthoughts"]:
        lex_path = REPO_ROOT / "data" / "processed" / f"{project}_c_lex.jsonl"
        sem_path = REPO_ROOT / "data" / "processed" / f"{project}_c_sem.jsonl"
        lex_ids = {item["math500_id"] for item in load_jsonl(lex_path)} if lex_path.exists() else set()
        sem_ids = {item["math500_id"] for item in load_jsonl(sem_path)} if sem_path.exists() else set()
        total_ids = lex_ids | sem_ids

        n_lex, n_sem, n_tot = len(lex_ids), len(sem_ids), len(total_ids)
        ci_lex = proportion_bootstrap(n_lex, MATH500_SIZE, n_bootstrap=n_bootstrap, seed=seed)
        ci_sem = proportion_bootstrap(n_sem, MATH500_SIZE, n_bootstrap=n_bootstrap, seed=seed)
        ci_tot = proportion_bootstrap(n_tot, MATH500_SIZE, n_bootstrap=n_bootstrap, seed=seed)

        rows.append({
            "dataset": project,
            "filter": DATASET_LABELS[project],
            "n_lex": n_lex,
            "n_sem": n_sem,
            "n_total": n_tot,
            "pct_lex": round(n_lex / MATH500_SIZE * 100, 1),
            "pct_sem": round(n_sem / MATH500_SIZE * 100, 1),
            "pct_total": round(n_tot / MATH500_SIZE * 100, 1),
            "ci_lex": f"[{ci_lex[0] * 100:.1f}, {ci_lex[1] * 100:.1f}]",
            "ci_sem": f"[{ci_sem[0] * 100:.1f}, {ci_sem[1] * 100:.1f}]",
            "ci_total": f"[{ci_tot[0] * 100:.1f}, {ci_tot[1] * 100:.1f}]",
            "failure_mode": _classify_mode(n_lex, n_sem),
        })

    df = pd.DataFrame(rows)
    out = REPO_ROOT / "results" / "tables" / "failure_mode_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    _log.info("-> %s", out)


# ── tulu_sources ──────────────────────────────────────────────────────────────

TULU_SOURCE_LABELS = {
    "ai2-adapt-dev/numinamath_tir_math_decontaminated": "NuminaMath-TIR",
    "ai2-adapt-dev/personahub_math_v5_regen_149960": "PersonaHub-Math",
    "allenai/tulu-3-sft-personas-math-grade": "PersonaHub-Grade",
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k": "GSM8K",
    "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k": "PersonaHub-Algebra",
}


def tulu_sources() -> None:
    p = paths("tulu")
    id_to_source: dict[str, str] = {}
    source_sizes: dict[str, int] = defaultdict(int)
    for item in load_jsonl(p.train):
        src = item.get("source", "unknown")
        id_to_source[item["train_id"]] = src
        source_sizes[src] += 1
    _log.info("indexed %d Tulu items across %d sources", len(id_to_source), len(source_sizes))

    lex_by_source: dict[str, set[str]] = defaultdict(set)
    sem_by_source: dict[str, set[str]] = defaultdict(set)
    for item in load_jsonl(REPO_ROOT / "data" / "processed" / "tulu_c_lex.jsonl"):
        src = id_to_source.get(item["train_id"], "unknown")
        lex_by_source[src].add(item["math500_id"])
    for item in load_jsonl(REPO_ROOT / "data" / "processed" / "tulu_c_sem.jsonl"):
        src = id_to_source.get(item["train_id"], "unknown")
        sem_by_source[src].add(item["math500_id"])

    rows = []
    for src in sorted(source_sizes, key=lambda s: -source_sizes[s]):
        n_lex = len(lex_by_source[src])
        n_sem = len(sem_by_source[src])
        n_tot = len(lex_by_source[src] | sem_by_source[src])
        rows.append({
            "source": src,
            "label": TULU_SOURCE_LABELS.get(src, src),
            "train_size": source_sizes[src],
            "n_lex": n_lex,
            "n_sem": n_sem,
            "n_total": n_tot,
            "pct_lex": round(n_lex / MATH500_SIZE * 100, 1),
            "pct_sem": round(n_sem / MATH500_SIZE * 100, 1),
            "pct_total": round(n_tot / MATH500_SIZE * 100, 1),
        })

    df = pd.DataFrame(rows)
    out = REPO_ROOT / "results" / "tables" / "tulu_source_breakdown.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    _log.info("-> %s", out)


# ── ngram_sweep ───────────────────────────────────────────────────────────────

def ngram_sweep(min_shared: int) -> None:
    p = paths("openthoughts_full")
    train = load_jsonl(p.train)
    math500 = load_jsonl(p.math500)

    if not p.ngram_hits.exists():
        _log.error("base n=13 hits missing at %s — run 01_ngram_filter.py first", p.ngram_hits)
        return

    base_hits = load_jsonl(p.ngram_hits)
    rows = [_sweep_row(13, base_hits, min_shared)]

    for n in (15, 20):
        out = REPO_ROOT / "results" / "ngram" / f"openthoughts_full_hits_n{n}.jsonl"
        if out.exists():
            hits = load_jsonl(out)
            _log.info("n=%d cached: %d hits", n, len(hits))
        else:
            hits = find_hits(
                "openthoughts_full", train, math500,
                NgramConfig(n=n, threshold_mode="any"),
            )
            save_jsonl(hits, out)
        rows.append(_sweep_row(n, hits, min_shared))

    df = pd.DataFrame(rows)
    out_csv = REPO_ROOT / "results" / "tables" / "robustness_ngram_sweep.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    _log.info("-> %s", out_csv)


def _sweep_row(n: int, hits: list[dict], min_shared: int) -> dict:
    filtered = dedup_by_math500(hits, min_shared)
    raw_unique = len({h["math500_id"] for h in hits})
    return {
        "n": n,
        "raw_unique": raw_unique,
        f"filtered_unique_min{min_shared}": len(filtered),
        "pct_math500": round(len(filtered) / MATH500_SIZE * 100, 1),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", choices=("failure_mode", "tulu_sources", "ngram_sweep", "all"),
                        default="all")
    parser.add_argument("--config", default=str(_common.DEFAULT_CONFIG))
    parser.add_argument("--min-shared", type=int, default=5,
                        help="C_lex inclusion threshold (default: 5)")
    args = parser.parse_args()

    configure_logging()
    app_config = load_config(args.config)
    seed_everything(app_config.seed)

    if args.check in ("failure_mode", "all"):
        _log.info("=== failure_mode ===")
        failure_mode(n_bootstrap=app_config.stats.n_bootstrap, seed=app_config.stats.seed)
    if args.check in ("tulu_sources", "all"):
        _log.info("=== tulu_sources ===")
        tulu_sources()
    if args.check in ("ngram_sweep", "all"):
        _log.info("=== ngram_sweep ===")
        ngram_sweep(args.min_shared)


if __name__ == "__main__":
    main()
