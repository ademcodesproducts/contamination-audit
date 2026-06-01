"""
11_build_annotation_csv.py

Builds a CSV of all C_sem pairs for manual annotation.
Pre-fills human verdicts from prior worksheet annotations where available.

Output: results/csem_annotation.csv
"""

import jsonlines
import csv
from pathlib import Path

# Known verdicts from prior manual annotation (worksheet)
KNOWN_VERDICTS = {
    # Tülu top-20
    ("tulu", "math500_0386"): "Y",
    ("tulu", "math500_0378"): "Y",
    ("tulu", "math500_0371"): "Y",
    ("tulu", "math500_0053"): "N",
    ("tulu", "math500_0260"): "N",
    ("tulu", "math500_0077"): "N",
    ("tulu", "math500_0452"): "N",
    ("tulu", "math500_0300"): "Y",
    ("tulu", "math500_0435"): "Y",
    ("tulu", "math500_0288"): "N",
    ("tulu", "math500_0434"): "?",
    ("tulu", "math500_0244"): "Y",
    ("tulu", "math500_0263"): "N",
    ("tulu", "math500_0208"): "N",
    ("tulu", "math500_0489"): "Y",
    ("tulu", "math500_0034"): "N",
    ("tulu", "math500_0127"): "N",
    ("tulu", "math500_0473"): "?",
    ("tulu", "math500_0375"): "Y",
    ("tulu", "math500_0089"): "N",
    # OT top-15
    ("openthoughts", "math500_0033"): "Y",
    ("openthoughts", "math500_0053"): "N",
    ("openthoughts", "math500_0260"): "Y",
    ("openthoughts", "math500_0386"): "Y",
    ("openthoughts", "math500_0473"): "N",
    ("openthoughts", "math500_0431"): "N",
    ("openthoughts", "math500_0371"): "N",
    ("openthoughts", "math500_0043"): "N",
    ("openthoughts", "math500_0497"): "N",
    ("openthoughts", "math500_0077"): "N",
    ("openthoughts", "math500_0016"): "N",
    ("openthoughts", "math500_0434"): "Y",
    ("openthoughts", "math500_0303"): "?",
    ("openthoughts", "math500_0426"): "N",
    ("openthoughts", "math500_0305"): "N",
    # s1
    ("s1", "math500_0466"): "Y",
}

FILES = [
    "data/output/tulu_c_sem.jsonl",
    "data/output/openthoughts_c_sem.jsonl",
    "data/output/s1_c_sem.jsonl",
]

def main():
    rows = []
    seen = set()  # (project, math500_id) to deduplicate

    for path in FILES:
        with jsonlines.open(path) as r:
            items = list(r)

        # Sort by similarity descending, deduplicate by math500_id per project
        items.sort(key=lambda x: -x.get("similarity_score", 0))
        deduped = []
        seen_ids = set()
        for item in items:
            mid = item["math500_id"]
            if mid not in seen_ids:
                seen_ids.add(mid)
                deduped.append(item)

        project = items[0]["project"] if items else ""

        for item in deduped:
            mid = item["math500_id"]
            key = (project, mid)
            verdict = KNOWN_VERDICTS.get(key, "")
            rows.append({
                "dataset":          project,
                "math500_id":       mid,
                "subject":          item.get("math500_subject", ""),
                "level":            item.get("math500_level", ""),
                "similarity_score": round(item.get("similarity_score", 0), 3),
                "human_verdict":    verdict,
                "judge_reasoning":  item.get("reasoning", ""),
                "math500_problem":  item.get("math500_problem", "").replace("\n", " "),
                "train_problem":    item.get("train_problem", "").replace("\n", " "),
            })

    # Stats
    total     = len(rows)
    annotated = sum(1 for r in rows if r["human_verdict"] in ("Y","N","?"))
    remaining = total - annotated
    confirmed = sum(1 for r in rows if r["human_verdict"] == "Y")
    print(f"Total unique C_sem pairs: {total}")
    print(f"  Already annotated: {annotated}  (Y={confirmed}, N={sum(1 for r in rows if r['human_verdict']=='N')}, ?={sum(1 for r in rows if r['human_verdict']=='?')})")
    print(f"  Remaining blank:   {remaining}")

    out = Path("results/csem_annotation.csv")
    fields = ["dataset","math500_id","subject","level","similarity_score",
              "human_verdict","judge_reasoning","math500_problem","train_problem"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out}")
    print("Fill in 'human_verdict' column: Y=contaminated, N=false positive, ?=unsure")

if __name__ == "__main__":
    main()
