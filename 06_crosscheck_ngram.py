"""
Cross-check: which judge-CONTAMINATED pairs also appear in the 8-gram hits?

This tells you:
  - If a pair is in BOTH: 8-gram filter should have caught it but didn't → why?
    (likely tokenization/formatting differences between s1's internal copy and HF release)
  - If a pair is ONLY in judge results: genuine semantic contamination the filter missed
    → this is the paper's core finding

Prints a breakdown and saves results/crosscheck.tsv
"""

import pandas as pd
from pathlib import Path

OUT_DIR = Path("results")


def main():
    # Load judge results (CONTAMINATED only)
    judge_path = OUT_DIR / "judge_results.parquet"
    if not judge_path.exists():
        print("ERROR: judge_results.parquet not found. Run 04_llm_judge.py first.")
        return
    judge_df = pd.read_parquet(judge_path)
    contaminated = judge_df[judge_df["label"] == "CONTAMINATED"].copy()
    print(f"Judge-flagged CONTAMINATED pairs: {len(contaminated)}")

    # Load 8-gram hits
    ngram_path = OUT_DIR / "ngram_hits.tsv"
    if not ngram_path.exists():
        print("ERROR: ngram_hits.tsv not found. Run 02_ngram_baseline.py first.")
        return
    ngram_df = pd.read_csv(ngram_path, sep="\t")
    print(f"8-gram hits: {len(ngram_df)}")

    # Build a set of (s1k_id, math500_id) pairs from ngram hits
    ngram_pairs = set(zip(ngram_df["s1k_id"], ngram_df["math500_id"]))

    # Tag each contaminated pair
    contaminated["in_ngram_hits"] = contaminated.apply(
        lambda r: (r["s1k_id"], r["math500_id"]) in ngram_pairs, axis=1
    )

    # For pairs in ngram hits, get the overlap count
    ngram_lookup = dict(zip(
        zip(ngram_df["s1k_id"], ngram_df["math500_id"]),
        ngram_df["overlap_count"]
    ))
    contaminated["ngram_overlap_count"] = contaminated.apply(
        lambda r: ngram_lookup.get((r["s1k_id"], r["math500_id"]), 0), axis=1
    )

    # Split into categories
    caught_by_both = contaminated[contaminated["in_ngram_hits"]]
    semantic_only  = contaminated[~contaminated["in_ngram_hits"]]

    print(f"\n{'='*60}")
    print(f"CAUGHT BY BOTH (8-gram + judge):  {len(caught_by_both)}")
    print(f"  → These should have been filtered but weren't.")
    print(f"  → Likely cause: whitespace/punctuation changed tokenization.")
    print(f"  → NOT your paper's main finding (filter techincally flagged them).")
    print()
    print(f"SEMANTIC-ONLY (judge only, 8-gram missed): {len(semantic_only)}")
    print(f"  → These are your paper's core finding.")
    print(f"  → Problems are structurally equivalent but share no 8-gram token overlap.")

    print(f"\n{'='*60}")
    print("\nSEMANTIC-ONLY pairs (the key finding):")
    cols = ["s1k_id", "math500_id", "cosine_sim", "justification", "s1k_problem", "math500_problem", "math500_subject"]
    for _, row in semantic_only[cols].iterrows():
        print(f"\n  s1K#{int(row['s1k_id'])} <-> MATH-500#{int(row['math500_id'])} (sim={row['cosine_sim']:.3f}, {row['math500_subject']})")
        print(f"  s1K:     {str(row['s1k_problem'])[:120]}...")
        print(f"  MATH500: {str(row['math500_problem'])[:120]}...")
        print(f"  Judge:   {row['justification']}")

    print(f"\n{'='*60}")
    print("\nCAUGHT BY BOTH pairs (8-gram also hit, but why wasn't it filtered?):")
    cols2 = ["s1k_id", "math500_id", "cosine_sim", "ngram_overlap_count", "s1k_problem", "math500_problem"]
    for _, row in caught_by_both[cols2].iterrows():
        print(f"\n  s1K#{int(row['s1k_id'])} <-> MATH-500#{int(row['math500_id'])} "
              f"(sim={row['cosine_sim']:.3f}, {int(row['ngram_overlap_count'])} shared 8-grams)")
        print(f"  s1K:     {str(row['s1k_problem'])[:120]}...")
        print(f"  MATH500: {str(row['math500_problem'])[:120]}...")

    # Save full crosscheck table
    out = contaminated[[
        "s1k_id", "math500_id", "cosine_sim", "in_ngram_hits", "ngram_overlap_count",
        "label", "justification", "math500_subject", "s1k_problem", "math500_problem"
    ]]
    out.to_csv(OUT_DIR / "crosscheck.tsv", sep="\t", index=False)
    print(f"\nFull crosscheck saved to {OUT_DIR / 'crosscheck.tsv'}")


if __name__ == "__main__":
    main()
