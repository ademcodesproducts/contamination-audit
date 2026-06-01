"""
Step 3A: Embedding-based retrieval to find candidate pairs for LLM judging.

Embeds all s1K and MATH-500 problems with a sentence transformer,
then identifies (s1K, MATH-500) pairs above a cosine similarity threshold.
This narrows 500K pairs down to a manageable candidate set.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

# Similarity threshold: pairs above this go to the LLM judge
THRESHOLD = 0.75
# Also keep top-K per s1K item regardless of threshold (catches borderline cases)
TOP_K = 5


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    s1k = load_json(DATA_DIR / "s1k.json")
    math500 = load_json(DATA_DIR / "math500.json")
    print(f"Loaded {len(s1k)} s1K items, {len(math500)} MATH-500 items.")
    print(f"Will compute {len(s1k) * len(math500):,} pairs (embedding approach avoids LLM calls for all of them).")

    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    s1k_texts = [item["problem"] for item in s1k]
    math500_texts = [item["problem"] for item in math500]

    print("Encoding s1K problems...")
    s1k_embs = model.encode(s1k_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    print("Encoding MATH-500 problems...")
    math500_embs = model.encode(math500_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    # Save embeddings for reuse
    np.save(OUT_DIR / "s1k_embeddings.npy", s1k_embs)
    np.save(OUT_DIR / "math500_embeddings.npy", math500_embs)
    print("  Embeddings saved.")

    print("\nComputing cosine similarity matrix...")
    # Since embeddings are normalized, dot product = cosine similarity
    sim_matrix = s1k_embs @ math500_embs.T  # shape: (1000, 500)
    print(f"  Similarity matrix shape: {sim_matrix.shape}")

    # Collect candidates
    candidates = []
    for i in range(len(s1k)):
        row = sim_matrix[i]
        # Threshold-based
        above_thresh = np.where(row > THRESHOLD)[0]
        # Top-K
        top_k_idx = np.argsort(row)[-TOP_K:][::-1]
        # Union
        combined = set(above_thresh.tolist()) | set(top_k_idx.tolist())
        for j in combined:
            candidates.append(
                {
                    "s1k_id": s1k[i]["id"],
                    "math500_id": math500[j]["id"],
                    "cosine_sim": float(row[j]),
                    "above_threshold": bool(row[j] > THRESHOLD),
                    "s1k_problem": s1k[i]["problem"],
                    "math500_problem": math500[j]["problem"],
                    "s1k_solution": s1k[i].get("solution", ""),
                    "math500_solution": math500[j].get("solution", ""),
                    "s1k_source": s1k[i].get("source", ""),
                    "math500_subject": math500[j].get("subject", ""),
                }
            )

    # Deduplicate and sort by similarity
    df = pd.DataFrame(candidates).drop_duplicates(subset=["s1k_id", "math500_id"])
    df = df.sort_values("cosine_sim", ascending=False)

    out_path = OUT_DIR / "candidates.parquet"
    df.to_parquet(out_path, index=False)

    # Also save a human-readable TSV of high-confidence candidates
    high_conf = df[df["above_threshold"]]
    high_conf.to_csv(OUT_DIR / "candidates_above_threshold.tsv", sep="\t", index=False)

    print(f"\n{'='*50}")
    print(f"Total candidate pairs (threshold OR top-K): {len(df):,}")
    print(f"  Above threshold ({THRESHOLD}): {len(high_conf):,}")
    print(f"  Top-K only (below threshold): {len(df) - len(high_conf):,}")
    print(f"\nSimilarity distribution (above-threshold pairs):")
    if len(high_conf) > 0:
        print(high_conf["cosine_sim"].describe().to_string())
    print(f"\nResults saved to {out_path}")
    print("Next: run 04_llm_judge.py to classify candidates as CONTAMINATED/RELATED/CLEAN")


if __name__ == "__main__":
    main()
