"""
LLM-as-judge re-scorer using OpenAI GPT-4o-mini.
Judges every record in did_traces_openthoughts.jsonl for correctness.
For null final_answer, extracts the answer from the trace first.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Load API key from .env
env_path = os.path.join(os.path.dirname(__file__), ".env")
for line in open(env_path):
    k, _, v = line.strip().partition("=")
    if k == "OPENAI_API_KEY":
        os.environ["OPENAI_API_KEY"] = v

client = OpenAI()

JUDGE_PROMPT = """You are evaluating a math problem solution. Decide if the model's answer is correct given the ground truth.

Ground truth: {ground_truth}
Model answer: {model_answer}

Two answers are correct if they are mathematically equivalent, for example:
- 30 and 30° and 30^\\circ are the same
- 1/2 and 0.5 are the same
- 3\\sqrt{{13}} and 3*sqrt(13) are the same
- (-∞, 2) ∪ (3, ∞) and (-inf,2)U(3,inf) are the same
- Different LaTeX formatting of the same expression counts as correct

Reply with exactly one word: CORRECT or WRONG"""

EXTRACT_PROMPT = """A student solved a math problem. Extract their final answer from the solution below.
Return ONLY the final answer value, nothing else. If you cannot find a clear final answer, return NULL.

Solution:
{trace}"""


def extract_answer_from_trace(trace: str, math500_id: str) -> str | None:
    # First try regex for \boxed{}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', trace)
    if boxed:
        return boxed[-1].strip()
    # Fall back to GPT extraction
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=50,
            temperature=0,
            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(trace=trace[-3000:])}]
        )
        ans = resp.choices[0].message.content.strip()
        return None if ans.upper() == "NULL" else ans
    except Exception as e:
        print(f"    extraction error for {math500_id}: {e}")
        return None


def judge_record(rec: dict) -> dict:
    math500_id = rec["math500_id"]
    ground_truth = rec["ground_truth"]
    model_answer = rec.get("final_answer")

    # Extract from trace if no final answer
    extracted = False
    if model_answer is None:
        model_answer = extract_answer_from_trace(rec.get("full_trace", ""), math500_id)
        if model_answer:
            extracted = True

    if model_answer is None:
        return {**rec, "llm_correct": False, "llm_model_answer": None, "extracted": False}

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=5,
            temperature=0,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                ground_truth=ground_truth,
                model_answer=model_answer
            )}]
        )
        verdict = resp.choices[0].message.content.strip().upper()
        is_correct = "CORRECT" in verdict
    except Exception as e:
        print(f"    judge error for {math500_id}: {e}")
        is_correct = False

    return {**rec, "llm_correct": is_correct, "llm_model_answer": model_answer, "extracted": extracted}


def main():
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results/did_traces_openthoughts.jsonl"
    with open(path, encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    print(f"Judging {len(records)} records with GPT-4o-mini...\n")

    judged = [None] * len(records)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(judge_record, rec): i for i, rec in enumerate(records)}
        for future in as_completed(futures):
            i = futures[future]
            rec = records[i]
            result = future.result()
            judged[i] = result
            orig = "Y" if rec["correct"] else "N"
            new = "Y" if result["llm_correct"] else "N"
            flag = " *** FLIP" if rec["correct"] != result["llm_correct"] else ""
            extr = " [extracted]" if result["extracted"] else ""
            ans = result["llm_model_answer"] or "null"
            safe_ans = ans.encode('ascii', 'replace').decode('ascii')
            print(f"  [{i+1:3d}] {rec['math500_id']} {rec['split']:12} {rec['perturbation_type']:14} "
                  f"orig={orig} llm={new}{flag}{extr}  answer={safe_ans:.40s}")

    # Build corrected table
    cells = {}
    flips_up = []
    flips_down = []
    for r in judged:
        key = (r["split"], r["perturbation_type"])
        cells.setdefault(key, {"total": 0, "orig_correct": 0, "llm_correct": 0})
        cells[key]["total"] += 1
        if r["correct"]:
            cells[key]["orig_correct"] += 1
        if r["llm_correct"]:
            cells[key]["llm_correct"] += 1
        if not r["correct"] and r["llm_correct"]:
            flips_up.append(r)
        if r["correct"] and not r["llm_correct"]:
            flips_down.append(r)

    print("\n" + "="*70)
    print(f"{'Cell':<35} {'Original':>10} {'LLM judge':>10} {'Delta':>7}")
    print("="*70)
    order = [
        ("contaminated", "original"),
        ("contaminated", "number_swap"),
        ("contaminated", "surface_noise"),
        ("clean", "original"),
        ("clean", "number_swap"),
        ("clean", "surface_noise"),
    ]
    for key in order:
        if key in cells:
            c = cells[key]
            label = f"{key[0]} / {key[1]}"
            orig = f"{c['orig_correct']}/{c['total']}"
            llm = f"{c['llm_correct']}/{c['total']}"
            delta = c["llm_correct"] - c["orig_correct"]
            sign = f"+{delta}" if delta > 0 else str(delta)
            print(f"  {label:<33} {orig:>10} {llm:>10} {sign:>5}")

    print(f"\nFlips wrong->correct: {len(flips_up)}")
    for r in flips_up:
        ans = (r['llm_model_answer'] or '').encode('ascii','replace').decode('ascii')
        gt = (r['ground_truth'] or '').encode('ascii','replace').decode('ascii')
        print(f"  {r['math500_id']} {r['split']}/{r['perturbation_type']}  answer={ans!r}  gt={gt!r}")

    print(f"\nFlips correct->wrong: {len(flips_down)}")
    for r in flips_down:
        ans = (r['llm_model_answer'] or '').encode('ascii','replace').decode('ascii')
        gt = (r['ground_truth'] or '').encode('ascii','replace').decode('ascii')
        print(f"  {r['math500_id']} {r['split']}/{r['perturbation_type']}  answer={ans!r}  gt={gt!r}")

    # Save results
    out_path = path.replace(".jsonl", "_judged.jsonl")
    with open(out_path, "w") as f:
        for r in judged:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
