"""Fetch the corpus and benchmarks the paper's results are computed over.

Everything written here is derived from public HuggingFace datasets, so the files
are gitignored and regenerated rather than committed. The split between the two
benchmark groups is the paper's central design and is defined here:

  measurement -- competitions held on or before September 2025, so they could
                 plausibly be inside a corpus assembled by October 2025.
  control     -- competitions held January 2026 onward, so they provably cannot
                 be. Any flag against these is a false positive by construction.

Olmo 3 was announced 2025-11-20; we treat Dolci as closed by roughly October 2025
and leave an ambiguous band (Oct--Dec 2025) out of both groups.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import _common  # noqa: F401

from contamination_audit.config import configure_logging

_log = logging.getLogger("prepare_paper_data")

RAW = Path("data/raw")

MEASUREMENT = [
    "apex_2025", "apex-shortlist", "smt_2025", "cmimc_2025", "brumo_2025",
    "aime_2025", "hmmt_feb_2025", "aime_2024_I", "aime_2024_II", "hmmt_feb_2024",
]
CONTROL = [
    "aime_2026", "hmmt_feb_2026", "usamo_2026",
    "arxivmath-0126", "arxivmath-0326", "arxivmath-0626",
    "brokenarxiv-0326", "brokenarxiv-0626",
]


def pull_olymmath() -> None:
    from datasets import load_dataset
    for cfg in ["en-hard", "en-easy"]:
        tag = cfg.replace("-", "_")
        out = RAW / f"olymmath_{tag}.jsonl"
        ds = load_dataset("RUC-AIBOX/OlymMATH", cfg, split="test")
        with out.open("w", encoding="utf-8") as f:
            for i, r in enumerate(ds):
                f.write(json.dumps({
                    "id": f"olym_{tag}_{i:04d}",
                    "problem": r.get("problem") or r.get("question", ""),
                    "answer": str(r.get("answer", "")),
                }) + "\n")
        _log.info("OlymMATH %s: %d items -> %s", cfg, len(ds), out)


def pull_matharena(names: list[str], group: str, out: Path) -> None:
    from datasets import load_dataset
    rows, failed = [], []
    for nm in names:
        try:
            ds = load_dataset(f"MathArena/{nm}", split="train")
        except Exception as e:  # noqa: BLE001
            failed.append((nm, str(e)[:60]))
            continue
        if "problem" not in ds.column_names:
            failed.append((nm, f"no problem field: {ds.column_names}"))
            continue
        for i, r in enumerate(ds):
            rows.append({"id": f"{nm}_{i:04d}", "benchmark": nm, "group": group,
                         "problem": r["problem"], "answer": str(r.get("answer", ""))})
        _log.info("  %-22s %4d", nm, len(ds))
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    _log.info("%s: %d items -> %s", group, len(rows), out)
    for nm, err in failed:
        _log.error("  FAILED %s: %s", nm, err)


def _first_user(msgs) -> str:
    for m in msgs or []:
        if (m.get("role") or m.get("from")) in ("user", "human"):
            c = m.get("content") or m.get("value") or ""
            if isinstance(c, str):
                return c
            return " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return ""


def pull_dolci(target: int) -> None:
    """Stream Dolci, keeping only the first user turn -- all a filter reads.

    Streaming drops connections on long pulls, so this resumes by skipping what
    it already has rather than restarting.
    """
    from datasets import load_dataset
    out = RAW / "dolci_questions.jsonl"
    seen: set[str] = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                seen.add(json.loads(line)["train_id"])
            except Exception:  # noqa: BLE001
                pass
        _log.info("resuming with %d rows already on disk", len(seen))

    n = len(seen)
    with out.open("a", encoding="utf-8") as f:
        for attempt in range(30):
            if n >= target:
                break
            try:
                ds = load_dataset("allenai/Dolci-Think-SFT-32B", split="train",
                                  streaming=True)
                if n:
                    ds = ds.skip(n)
                for row in ds:
                    rid = row.get("id")
                    if rid in seen:
                        continue
                    q = _first_user(row.get("messages"))
                    if not q.strip():
                        continue
                    seen.add(rid)
                    f.write(json.dumps({"train_id": rid, "problem": q,
                                        "source": row.get("source")}) + "\n")
                    n += 1
                    if n % 50_000 == 0:
                        f.flush()
                        _log.info("  %d rows", n)
                    if n >= target:
                        break
            except Exception as e:  # noqa: BLE001
                _log.warning("  stream dropped (%s), retry %d", type(e).__name__, attempt + 1)
                time.sleep(5)
    _log.info("Dolci: %d rows -> %s", n, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-rows", type=int, default=390_000,
                    help="the paper's largest scaling point")
    ap.add_argument("--skip-corpus", action="store_true")
    args = ap.parse_args()
    configure_logging()
    RAW.mkdir(parents=True, exist_ok=True)

    pull_olymmath()
    _log.info("measurement set (competitions <= Sept 2025):")
    pull_matharena(MEASUREMENT, "measure", RAW / "bench_measure.jsonl")
    _log.info("control set (competitions Jan 2026 onward):")
    pull_matharena(CONTROL, "control", RAW / "bench_control.jsonl")

    if not args.skip_corpus:
        pull_dolci(args.corpus_rows)


if __name__ == "__main__":
    main()
