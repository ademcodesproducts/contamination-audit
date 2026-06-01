"""
Step 4: Analyze judge results and produce the pilot report.

Generates:
  - results/pilot_report.md  — narrative summary with the 4 key numbers
  - results/manual_review_sample.tsv  — 30 CONTAMINATED pairs for hand validation
  - results/precision_worksheet.tsv  — blank sheet to fill in manual labels
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

OUT_DIR = Path("results")
MANUAL_SAMPLE_SIZE = 30


def load_parquet(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run previous steps first.")
    return pd.read_parquet(path)


def main():
    print("Loading results...")
    judge_df = load_parquet(OUT_DIR / "judge_results.parquet")

    # -- Core numbers --
    n_candidates = len(judge_df)
    n_contaminated = (judge_df["label"] == "CONTAMINATED").sum()
    n_related = (judge_df["label"] == "RELATED").sum()
    n_clean = (judge_df["label"] == "CLEAN").sum()
    n_errors = (judge_df["label"] == "ERROR").sum()

    # Unique s1K items flagged as contaminated
    contaminated_pairs = judge_df[judge_df["label"] == "CONTAMINATED"]
    unique_s1k_contaminated = contaminated_pairs["s1k_id"].nunique()
    unique_math500_contaminated = contaminated_pairs["math500_id"].nunique()

    # Similarity distribution of contaminated pairs
    if len(contaminated_pairs) > 0:
        sim_stats = contaminated_pairs["cosine_sim"].describe()
    else:
        sim_stats = None

    # -- Manual review sample (CONTAMINATED first, then RELATED) --
    flagged = judge_df[judge_df["label"].isin(["CONTAMINATED", "RELATED"])].sort_values(
        ["label", "cosine_sim"], ascending=[True, False]  # CONTAMINATED first
    )
    sample_size = min(MANUAL_SAMPLE_SIZE, len(flagged))
    sample = flagged.head(sample_size).copy()

    sample_out = sample[
        ["s1k_id", "math500_id", "cosine_sim", "label",
         "q1_same_reasoning", "q2_unfair_advantage", "q3_structurally_equiv",
         "justification", "s1k_problem", "math500_problem",
         "s1k_source", "math500_subject"]
    ]
    sample_out.to_csv(OUT_DIR / "manual_review_sample.tsv", sep="\t", index=False)

    # Precision worksheet — blank column for human label
    ws = sample_out.copy()
    ws["human_label"] = ""  # fill in: CONTAMINATED / RELATED / CLEAN / UNCLEAR
    ws["human_notes"] = ""
    ws.to_csv(OUT_DIR / "precision_worksheet.tsv", sep="\t", index=False)

    # -- Similarity histogram buckets --
    bins = [0, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]
    labels_bins = ["<0.60", "0.60-0.65", "0.65-0.70", "0.70-0.75",
                   "0.75-0.80", "0.80-0.85", "0.85-0.90", "≥0.90"]
    judge_df["sim_bucket"] = pd.cut(judge_df["cosine_sim"], bins=bins, labels=labels_bins, right=False)
    sim_by_label = judge_df.groupby(["sim_bucket", "label"], observed=True).size().unstack(fill_value=0)

    # -- Subject breakdown --
    subject_breakdown = (
        contaminated_pairs.groupby("math500_subject")["math500_id"]
        .nunique()
        .sort_values(ascending=False)
        if len(contaminated_pairs) > 0 and "math500_subject" in contaminated_pairs.columns
        else pd.Series(dtype=int)
    )

    # -- Build report --
    report_lines = [
        "# Pilot Report: Semantic Contamination in s1K vs MATH-500",
        "",
        "## Key Numbers",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Candidate pairs (from embedding retrieval) | {n_candidates:,} |",
        f"| Pairs labeled CONTAMINATED by judge | {n_contaminated} |",
        f"| Pairs labeled RELATED by judge | {n_related} |",
        f"| Pairs labeled CLEAN by judge | {n_clean} |",
        f"| Errors / unparseable | {n_errors} |",
        f"| Unique s1K items flagged CONTAMINATED | {unique_s1k_contaminated} |",
        f"| Unique MATH-500 items with a CONTAMINATED s1K pair | {unique_math500_contaminated} |",
        "",
        "## Interpretation",
        "",
    ]

    if n_contaminated >= 50:
        verdict = (
            f"**Strong signal.** {n_contaminated} CONTAMINATED pairs found "
            f"({unique_s1k_contaminated} unique s1K items). This exceeds the >30–50 threshold "
            f"for a non-trivial finding. If manual validation confirms >70% precision, "
            f"the paper premise holds."
        )
    elif n_contaminated >= 20:
        verdict = (
            f"**Moderate signal.** {n_contaminated} CONTAMINATED pairs found "
            f"({unique_s1k_contaminated} unique s1K items). This is in the borderline range. "
            f"Manual validation is critical — if precision is high, the finding is publishable "
            f"as a targeted audit; if precision is low, revisit the judge prompt or threshold."
        )
    elif n_contaminated > 0:
        verdict = (
            f"**Weak signal.** Only {n_contaminated} CONTAMINATED pairs found. "
            f"This is below the >20 threshold for a compelling finding. Consider: "
            f"(1) lowering the cosine similarity threshold, (2) improving the judge prompt, "
            f"or (3) reconsidering the dataset pair (try OpenThoughts-114K vs MATH-500)."
        )
    else:
        verdict = (
            "**No contamination found.** The LLM judge found 0 CONTAMINATED pairs. "
            "The paper premise does not hold for this dataset pair. "
            "Recommend switching focus to a larger training set (e.g., OpenThoughts-114K) "
            "or a different benchmark."
        )

    report_lines += [verdict, ""]

    report_lines += [
        "## Similarity Distribution by Label",
        "",
        "```",
        str(sim_by_label),
        "```",
        "",
    ]

    if len(subject_breakdown) > 0:
        report_lines += [
            "## CONTAMINATED Pairs by MATH-500 Subject",
            "",
            "```",
            str(subject_breakdown),
            "```",
            "",
        ]

    report_lines += [
        "## Next Steps",
        "",
        "1. **Manual validation**: Open `results/precision_worksheet.tsv`, review each pair, "
        "   fill in `human_label` and `human_notes`. Target: 20–30 cases.",
        "2. **Compute precision**: precision = (human CONTAMINATED) / (judge CONTAMINATED) in your sample.",
        "3. **If precision ≥ 70% and n_contaminated ≥ 20**: write up findings.",
        "4. **If precision < 60%**: revise judge prompt (make q2/q3 stricter) and rerun.",
        "5. **Scale up**: if pilot succeeds, run on OpenThoughts-114K for the full paper.",
        "",
        "## Files Generated",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `results/judge_results.parquet` | All judged pairs with labels |",
        "| `results/flagged_pairs.tsv` | Human-readable CONTAMINATED+RELATED pairs |",
        "| `results/manual_review_sample.tsv` | 30-pair sample for manual review |",
        "| `results/precision_worksheet.tsv` | Blank worksheet for precision estimation |",
    ]

    report = "\n".join(report_lines)
    report_path = OUT_DIR / "pilot_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\n{'='*50}")
    print(f"Full report saved to {report_path}")
    print(f"Manual review sample: {OUT_DIR / 'manual_review_sample.tsv'} ({sample_size} pairs)")
    print(f"Precision worksheet:  {OUT_DIR / 'precision_worksheet.tsv'}")


if __name__ == "__main__":
    main()
