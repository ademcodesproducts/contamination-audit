"""
Step 2: Replicate s1's 8-gram decontamination filter.

Since s1K was already filtered, we expect near-zero hits — confirming we've
replicated their method correctly. Outputs a TSV of any hits found.
"""

import json
import csv
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

N = 8  # n-gram size used by s1


def get_ngrams(tokens: list[int], n: int = 8) -> set[tuple]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    print("Loading tokenizer (Qwen2-7B-Instruct)...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        use_hf = True
    except Exception as e:
        print(f"  HuggingFace tokenizer unavailable ({e}), falling back to whitespace tokenizer.")
        use_hf = False

    def tokenize(text: str) -> list:
        if use_hf:
            return tokenizer.encode(text, add_special_tokens=False)
        # Fallback: character-level 8-grams on lowercased text (less accurate but reproducible)
        return list(text.lower())

    s1k = load_json(DATA_DIR / "s1k.json")
    math500 = load_json(DATA_DIR / "math500.json")
    print(f"Loaded {len(s1k)} s1K items, {len(math500)} MATH-500 items.")

    print("Computing MATH-500 n-gram sets...")
    math500_ngrams = [
        get_ngrams(tokenize(item["problem"]), N)
        for item in tqdm(math500, desc="MATH-500")
    ]

    hits = []
    print(f"\nScanning s1K × MATH-500 for {N}-gram overlaps...")
    for s1_item in tqdm(s1k, desc="s1K"):
        s1_tokens = tokenize(s1_item["problem"])
        s1_ngrams = get_ngrams(s1_tokens, N)
        for j, m_ngrams in enumerate(math500_ngrams):
            overlap = s1_ngrams & m_ngrams
            if overlap:
                hits.append(
                    {
                        "s1k_id": s1_item["id"],
                        "math500_id": math500[j]["id"],
                        "overlap_count": len(overlap),
                        "s1k_problem": s1_item["problem"][:200],
                        "math500_problem": math500[j]["problem"][:200],
                    }
                )

    out_path = OUT_DIR / "ngram_hits.tsv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=hits[0].keys() if hits else [], delimiter="\t")
        if hits:
            writer.writeheader()
            writer.writerows(hits)

    print(f"\n{'='*50}")
    print(f"8-gram hits found: {len(hits)}")
    print(f"Expected: ~0 (s1K was already decontaminated by this method)")
    print(f"Results saved to {out_path}")

    if len(hits) > 0:
        print("\nSample hits (first 3):")
        for h in hits[:3]:
            print(f"  s1K#{h['s1k_id']} <-> MATH-500#{h['math500_id']} ({h['overlap_count']} shared {N}-grams)")


if __name__ == "__main__":
    main()
