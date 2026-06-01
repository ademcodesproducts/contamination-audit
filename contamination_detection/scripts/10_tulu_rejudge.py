"""
10_tulu_rejudge.py

Re-runs the LLM judge on Tulu candidates using a stricter prompt.

The original prompt was too permissive: it flagged "same theorem used" as CONTAMINATED.
The stricter prompt requires that memorizing the training solution would directly answer
the benchmark problem -- same solution steps, not just same topic.

Runs on all candidates at sim>=0.80 (174 pairs), which is the validated threshold.

Output:
  results/judge_results/tulu_judge_results_v2.jsonl
  results/tulu_csem_v2_worksheet.txt
  data/output/tulu_c_sem.jsonl  (updated with confirmed items)
"""

from openai import OpenAI
import jsonlines
import os
import json
import time
import re
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

SIM_THRESHOLD = 0.80
OUTPUT_PATH   = "results/judge_results/tulu_judge_results_v2.jsonl"
CANDIDATES    = "results/embedding_candidates/tulu_candidates.jsonl"

JUDGE_PROMPT = """\
You are an expert mathematician evaluating potential training data contamination in LLM benchmarks.

You will be shown two math problems: one from a training dataset and one from a test benchmark.

YOUR TASK: Determine whether a model that memorized the training problem's solution would have
a DIRECT, STEP-FOR-STEP advantage on the benchmark problem.

CLASSIFICATION RULES:

CONTAMINATED — mark this if and only if:
  (a) Near-paraphrase: the benchmark problem is the training problem with only minor surface changes
      (word order swap, synonym substitution, punctuation, variable renaming). The answer is the same.
  (b) Template substitution: the two problems use LITERALLY THE SAME formula or procedure, and ONLY
      the input constants differ (e.g., different coordinates for the same dot-product angle formula,
      different starting values for the same geometric sequence formula). A student who copied the
      training solution's method verbatim would get the benchmark answer correct.

RELATED — mark this if:
  The problems use the same general technique or theorem, but the specific computation differs
  enough that the training solution does not directly transfer. Examples:
  - Same function type (arcsin, GCD, base conversion) but different values requiring different work
  - Same area (e.g., polygon angles) but different specific relationship or question
  - Same general approach but the student would need to re-derive rather than copy

CLEAN — mark this if:
  The similarity is superficial. The problems are genuinely independent despite similar keywords.

CRITICAL DISTINCTION:
  "Same topic" or "same theorem mentioned" is NOT sufficient for CONTAMINATED.
  Ask yourself: if a student copied the training solution word-for-word, would they answer the
  benchmark correctly? If yes -> CONTAMINATED. If they would need to re-derive -> RELATED.

EXAMPLES:
  CONTAMINATED: "How many integers in |x-2|<=5.6?" vs "How many integers are there in |x-2|<=5.6?"
    (word-order swap only, same answer)
  CONTAMINATED: angle ABC with coords (1,-11,2),(3,-4,1),(-2,1,-1) vs same formula, different coords
    (identical dot-product steps, only plug-in values change)
  RELATED: arcsin(-1/2) vs arcsin(1/sqrt(2))
    (same function, but different known values requiring different quadrant reasoning)
  RELATED: GCD(3339,2961,1491) vs GCD(9118,12173,33182)
    (same algorithm, but completely different computation)

Training problem:
{train_problem}

Benchmark problem:
{math500_problem}

Respond with a JSON object only, no other text:
{{"classification": "CONTAMINATED" or "RELATED" or "CLEAN", "confidence": "HIGH" or "MEDIUM" or "LOW", "reasoning": "one sentence explaining the classification", "shared_insight": "if CONTAMINATED, the specific shared formula or procedure; else null"}}"""


def make_client():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)


def judge_pair(client, train_problem, math500_problem, max_retries=3):
    prompt = JUDGE_PROMPT.format(
        train_problem=train_problem[:1500],
        math500_problem=math500_problem[:1500],
    )
    last_err = "unknown"
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
                timeout=30,
            )
            raw = resp.choices[0].message.content or ""
            raw_clean = re.sub(r"```(?:json)?\s*", "", raw).strip()
            try:
                return json.loads(raw_clean)
            except json.JSONDecodeError:
                pass
            m = re.search(r"\{.*\}", raw_clean, re.DOTALL)
            if m:
                candidate = m.group()
                candidate = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', candidate)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
            cls_match = re.search(r'"classification"\s*:\s*"(CONTAMINATED|RELATED|CLEAN)"', raw, re.IGNORECASE)
            conf_match = re.search(r'"confidence"\s*:\s*"(HIGH|MEDIUM|LOW)"', raw, re.IGNORECASE)
            reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', raw)
            if cls_match:
                return {
                    "classification": cls_match.group(1).upper(),
                    "confidence": conf_match.group(1).upper() if conf_match else "LOW",
                    "reasoning": reason_match.group(1) if reason_match else "extracted via regex",
                    "shared_insight": None,
                }
            last_err = f"no JSON in: {raw[:100]}"
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    print(f"\n  [WARN] Judge failed: {last_err}")
    return {"classification": "ERROR", "confidence": "LOW",
            "reasoning": last_err[:200], "shared_insight": None}


def build_worksheet(contaminated, output_path):
    lines = []
    lines.append("=" * 70)
    lines.append("  TULU v2 -- C_sem candidates for manual validation (stricter judge)")
    lines.append("  Mark each: Y=contaminated  N=false positive  ?=unsure")
    lines.append("=" * 70)
    lines.append("")
    for i, item in enumerate(contaminated, 1):
        lines.append(f"[{i}] sim={item['similarity_score']:.3f}  {item['math500_id']}  conf={item.get('confidence','?')}  verdict: ")
        lines.append(f"  MATH-500 : {item['math500_problem'][:200]}")
        lines.append(f"  TRAIN    : {item['train_problem'][:200]}")
        lines.append(f"  Judge    : {item.get('reasoning', '')}")
        lines.append("")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved worksheet: {output_path}")


def main():
    print("Loading Tulu candidates...")
    with jsonlines.open(CANDIDATES) as r:
        all_candidates = list(r)

    candidates = [c for c in all_candidates if c["similarity_score"] >= SIM_THRESHOLD]
    candidates.sort(key=lambda x: -x["similarity_score"])
    print(f"Candidates at sim>={SIM_THRESHOLD}: {len(candidates)}")

    # Resume support
    already_judged = {}
    if Path(OUTPUT_PATH).exists():
        with jsonlines.open(OUTPUT_PATH) as r:
            for item in r:
                already_judged[(item["math500_id"], item["train_id"])] = item
        print(f"Resuming: {len(already_judged)} already judged")

    client = make_client()
    results = list(already_judged.values())
    to_judge = [c for c in candidates if (c["math500_id"], c["train_id"]) not in already_judged]
    print(f"Remaining to judge: {len(to_judge)}")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    for candidate in tqdm(to_judge, desc="Re-judging Tulu (v2)"):
        judgment = judge_pair(client, candidate["train_problem"], candidate["math500_problem"])
        result = {**candidate, **judgment,
                  "contamination_type": (
                      "c_sem" if judgment["classification"] == "CONTAMINATED"
                      else "related" if judgment["classification"] == "RELATED"
                      else "clean_candidate"
                  )}
        results.append(result)
        with jsonlines.open(OUTPUT_PATH, mode="w") as w:
            w.write_all(results)
        time.sleep(0.1)

    confirmed   = [r for r in results if r["classification"] == "CONTAMINATED"]
    related     = [r for r in results if r["classification"] == "RELATED"]
    clean       = [r for r in results if r["classification"] == "CLEAN"]
    total_valid = len([r for r in results if r["classification"] != "ERROR"])

    print(f"\nResults (stricter prompt, sim>={SIM_THRESHOLD}):")
    print(f"  CONTAMINATED: {len(confirmed)}")
    print(f"  RELATED:      {len(related)}")
    print(f"  CLEAN:        {len(clean)}")
    print(f"  Judge precision: {len(confirmed)/max(total_valid,1)*100:.1f}%")

    # Build annotation worksheet
    build_worksheet(
        sorted(confirmed, key=lambda x: -x["similarity_score"]),
        "results/tulu_csem_v2_worksheet.txt"
    )

    # Update tulu_c_sem.jsonl with new confirmed items
    # (preserve existing fields expected by downstream scripts)
    output_items = []
    for item in confirmed:
        output_items.append({
            "project":           "tulu",
            "contamination_type": "c_sem",
            "math500_id":        item["math500_id"],
            "math500_problem":   item["math500_problem"],
            "math500_answer":    item.get("math500_answer", ""),
            "math500_subject":   item.get("math500_subject", ""),
            "math500_level":     item.get("math500_level", ""),
            "train_id":          item["train_id"],
            "train_problem":     item["train_problem"],
            "train_solution":    item.get("train_solution", ""),
            "similarity_score":  item["similarity_score"],
            "ngram_overlap":     item.get("ngram_overlap", 0),
            "classification":    item["classification"],
            "confidence":        item.get("confidence", ""),
            "reasoning":         item.get("reasoning", ""),
            "shared_insight":    item.get("shared_insight", ""),
        })

    with jsonlines.open("data/output/tulu_c_sem.jsonl", mode="w") as w:
        w.write_all(output_items)

    unique_ids = {i["math500_id"] for i in output_items}
    print(f"\nUpdated data/output/tulu_c_sem.jsonl: {len(output_items)} pairs, {len(unique_ids)} unique MATH-500 items")
    print("\nNext: annotate results/tulu_csem_v2_worksheet.txt, then re-run 09_build_canonical_list.py")


if __name__ == "__main__":
    main()
