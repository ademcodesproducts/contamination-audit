"""Stage 13: build the manual-annotation CSV of C_sem pairs.

Joins the deduped C_sem outputs from stage 5 with the prior manual verdicts in
data/processed/manual_verdicts.csv so an annotator only sees rows still needing
review. Y/N/? mean confirmed / false-positive / unsure.

Reads:  data/processed/{project}_c_sem.jsonl, data/processed/manual_verdicts.csv
Writes: results/annotations/csem_annotation.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import _common  # noqa: F401

from contamination_audit.config import configure_logging
from contamination_audit.io import REPO_ROOT, load_jsonl

_log = logging.getLogger("build_annotation_csv")


def _load_verdicts(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        _log.warning("no prior verdicts at %s; CSV will be unannotated", path)
        return {}
    verdicts: dict[tuple[str, str], str] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            verdicts[(row["project"], row["math500_id"])] = row["human_verdict"]
    return verdicts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdicts",
                        default=str(REPO_ROOT / "data" / "processed" / "manual_verdicts.csv"))
    parser.add_argument("--out",
                        default=str(REPO_ROOT / "results" / "annotations" / "csem_annotation.csv"))
    args = parser.parse_args()

    configure_logging()

    verdicts = _load_verdicts(Path(args.verdicts))

    rows = []
    for project in ("tulu", "openthoughts", "s1"):
        path = REPO_ROOT / "data" / "processed" / f"{project}_c_sem.jsonl"
        if not path.exists():
            continue
        items = sorted(load_jsonl(path), key=lambda x: -x.get("similarity_score", 0))
        seen: set[str] = set()
        for item in items:
            mid = item["math500_id"]
            if mid in seen:
                continue
            seen.add(mid)
            rows.append({
                "dataset": project,
                "math500_id": mid,
                "subject": item.get("math500_subject", ""),
                "level": item.get("math500_level", ""),
                "similarity_score": round(item.get("similarity_score", 0), 3),
                "human_verdict": verdicts.get((project, mid), ""),
                "judge_reasoning": item.get("reasoning", ""),
                "math500_problem": item.get("math500_problem", "").replace("\n", " "),
                "train_problem": item.get("train_problem", "").replace("\n", " "),
            })

    annotated = sum(1 for r in rows if r["human_verdict"] in ("Y", "N", "?"))
    confirmed = sum(1 for r in rows if r["human_verdict"] == "Y")
    _log.info("total pairs: %d  annotated: %d (Y=%d)  remaining: %d",
              len(rows), annotated, confirmed, len(rows) - annotated)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "math500_id", "subject", "level", "similarity_score",
              "human_verdict", "judge_reasoning", "math500_problem", "train_problem"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _log.info("wrote %s", out)


if __name__ == "__main__":
    main()
