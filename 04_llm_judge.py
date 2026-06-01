"""
Step 3B: LLM judge — classify candidate pairs as CONTAMINATED / RELATED / CLEAN.

Reads candidates from 03_embedding_retrieval.py output.
Uses OpenAI-compatible API (GPT-4o-mini by default; swap base_url for local models).
Costs roughly $0.01-0.02 per 100 pairs with GPT-4o-mini.

Usage:
    OPENAI_API_KEY=sk-... python 04_llm_judge.py
    OPENAI_API_KEY=sk-... python 04_llm_judge.py --only-above-threshold
    OPENAI_API_KEY=sk-... python 04_llm_judge.py --max-pairs 200  # for quick pilot
"""

import json
import time
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

OUT_DIR = Path("results")

JUDGE_PROMPT = """\
You are a strict judge evaluating potential benchmark contamination in math training data.

Problem A (from training data):
{s1k_problem}

Problem B (from benchmark):
{math500_problem}

Answer each question carefully. Be conservative — only answer Yes if you are confident.

1. Are these problems SPECIFICALLY equivalent — same setup, same constraints, same type of answer? \
Ignore surface rephrasing; also ignore mere topic overlap (e.g. "both involve triangles" or "both use AM-GM" is NOT enough for Yes). (Yes/No)

2. If a student had memorized the SPECIFIC solution to Problem A (not just the general technique), \
would that memorized solution give a direct, unfair advantage on Problem B — beyond what any student \
who knows the topic would have? (Yes/No)

3. Could the two problems share a solution template where plugging in Problem B's values into \
Problem A's solution approach yields the answer with minimal additional reasoning? (Yes/No)

Classify as:
- CONTAMINATED: 2 or more Yes answers (specific structural equivalence)
- RELATED: exactly 1 Yes answer (same topic/technique but independent problems)
- CLEAN: 0 Yes answers (distinct problems)

Respond in JSON with this exact format:
{{"q1": "Yes/No", "q2": "Yes/No", "q3": "Yes/No", "label": "CONTAMINATED/RELATED/CLEAN", "justification": "one sentence"}}
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--only-above-threshold", action="store_true",
                   help="Only judge pairs above the cosine similarity threshold (default: all candidates)")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Cap number of pairs to judge (for cost control)")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="Model to use (default: gpt-4o-mini)")
    p.add_argument("--base-url", default=None,
                   help="Custom base URL for OpenAI-compatible local inference")
    p.add_argument("--resume", action="store_true",
                   help="Skip pairs already judged in output file")
    return p.parse_args()


def call_judge(client, model, s1k_problem, math500_problem, max_retries=3):
    import re
    prompt = JUDGE_PROMPT.format(
        s1k_problem=s1k_problem[:1500],
        math500_problem=math500_problem[:1500],
    )
    last_error = "unknown"
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
                # No response_format — more compatible; we parse JSON from text
            )
            raw = resp.choices[0].message.content or ""
            # Try direct parse first
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
            # Try to extract {...} block
            match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            last_error = f"no JSON in response: {raw[:120]}"
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    # Surface the actual error so we can diagnose
    print(f"\n  [WARN] Judge failed after {max_retries} attempts: {last_error}")
    return {"q1": "Error", "q2": "Error", "q3": "Error",
            "label": "ERROR", "justification": last_error[:200]}


def main():
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Set it in .env or as environment variable.")
        print("  Example: OPENAI_API_KEY=sk-... python 04_llm_judge.py")
        return

    from openai import OpenAI
    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    # Load candidates
    candidates_path = OUT_DIR / "candidates.parquet"
    if not candidates_path.exists():
        print("ERROR: candidates.parquet not found. Run 03_embedding_retrieval.py first.")
        return

    df = pd.read_parquet(candidates_path)
    print(f"Loaded {len(df):,} candidate pairs.")

    if args.only_above_threshold:
        df = df[df["above_threshold"]]
        print(f"  Filtered to {len(df):,} above-threshold pairs.")

    df = df.sort_values("cosine_sim", ascending=False)

    if args.max_pairs:
        df = df.head(args.max_pairs)
        print(f"  Capped to {len(df):,} pairs (--max-pairs={args.max_pairs}).")

    out_path = OUT_DIR / "judge_results.parquet"
    already_done = set()
    results = []

    if args.resume and out_path.exists():
        existing = pd.read_parquet(out_path)
        already_done = set(zip(existing["s1k_id"], existing["math500_id"]))
        results = existing.to_dict("records")
        print(f"  Resuming: {len(already_done)} pairs already judged.")

    todo = df[~df.apply(lambda r: (r["s1k_id"], r["math500_id"]) in already_done, axis=1)]
    print(f"\nJudging {len(todo):,} pairs with {args.model}...")

    for _, row in tqdm(todo.iterrows(), total=len(todo)):
        verdict = call_judge(
            client, args.model,
            row["s1k_problem"],
            row["math500_problem"],
        )
        results.append(
            {
                "s1k_id": row["s1k_id"],
                "math500_id": row["math500_id"],
                "cosine_sim": row["cosine_sim"],
                "above_threshold": row["above_threshold"],
                "label": verdict.get("label", "ERROR"),
                "q1_same_reasoning": verdict.get("q1", ""),
                "q2_unfair_advantage": verdict.get("q2", ""),
                "q3_structurally_equiv": verdict.get("q3", ""),
                "justification": verdict.get("justification", ""),
                "s1k_problem": row["s1k_problem"],
                "math500_problem": row["math500_problem"],
                "s1k_source": row.get("s1k_source", ""),
                "math500_subject": row.get("math500_subject", ""),
            }
        )

    out_df = pd.DataFrame(results)
    out_df.to_parquet(out_path, index=False)

    # Also save human-readable TSV of flagged pairs
    flagged = out_df[out_df["label"].isin(["CONTAMINATED", "RELATED"])]
    flagged.to_csv(OUT_DIR / "flagged_pairs.tsv", sep="\t", index=False)

    print(f"\n{'='*50}")
    label_counts = out_df["label"].value_counts()
    print("Label distribution:")
    print(label_counts.to_string())
    print(f"\nTotal judged: {len(out_df):,}")
    print(f"CONTAMINATED: {label_counts.get('CONTAMINATED', 0)}")
    print(f"RELATED:      {label_counts.get('RELATED', 0)}")
    print(f"CLEAN:        {label_counts.get('CLEAN', 0)}")
    print(f"\nFlagged pairs saved to {OUT_DIR / 'flagged_pairs.tsv'}")
    print("Next: run 05_analyze_results.py for the full report")


if __name__ == "__main__":
    main()
