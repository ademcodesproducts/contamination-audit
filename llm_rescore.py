"""
LLM-as-judge re-scorer for DiD traces.
Re-scores correct=false records using Claude Haiku to check mathematical equivalence.
Also attempts to extract answers from null final_answer traces via regex.
"""

import json
import re
import os
import anthropic

client = anthropic.Anthropic()

JUDGE_PROMPT = """You are a math answer checker. Determine if the model answer is mathematically equivalent to the ground truth.

Ground truth: {ground_truth}
Model answer: {model_answer}

Consider these equivalent:
- Same number in different notation (30 == 30°, 0.5 == 1/2, sqrt(13)*3 == 3*sqrt(13))
- Same set/interval written differently
- Same vector with different LaTeX formatting
- Numerical values that round to the same number

Answer with exactly one word: YES or NO"""


def extract_from_trace(trace: str) -> str | None:
    """Try to extract a final answer from trace text when final_answer is null."""
    # Look for \boxed{...}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', trace)
    if boxed:
        return boxed[-1]
    # Look for "the answer is X" / "= X" at end of trace
    patterns = [
        r'(?:answer is|result is|value is)[:\s]+([^\n\.]+)',
        r'(?:therefore|thus|so)[,\s]+(?:the\s+)?(?:answer|value|result)\s+is[:\s]+([^\n\.]+)',
        r'\\boxed\{([^}]*)\}',
    ]
    for p in patterns:
        m = re.search(p, trace, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def judge_equivalence(model_answer: str, ground_truth: str) -> bool:
    """Call Claude Haiku to judge if model_answer is equivalent to ground_truth."""
    prompt = JUDGE_PROMPT.format(
        ground_truth=ground_truth,
        model_answer=model_answer
    )
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        messages=[{"role": "user", "content": prompt}]
    )
    verdict = resp.content[0].text.strip().upper()
    return verdict.startswith("YES")


def rescore_file(path: str) -> dict:
    with open(path) as f:
        records = [json.loads(line) for line in f]

    results = {}  # (split, perturbation_type) -> {"total": n, "correct": n, "rescored": n}

    for i, rec in enumerate(records):
        key = (rec["split"], rec["perturbation_type"])
        if key not in results:
            results[key] = {"total": 0, "correct": 0, "rescored": 0}
        results[key]["total"] += 1

        if rec["correct"]:
            results[key]["correct"] += 1
            continue

        # Try to get a model answer to judge
        model_answer = rec.get("final_answer")
        if model_answer is None:
            model_answer = extract_from_trace(rec.get("full_trace", ""))

        if model_answer is None:
            # No answer to judge — stays incorrect
            print(f"  [{i+1}] {rec['math500_id']} {key} — null, no extraction")
            continue

        ground_truth = rec["ground_truth"]
        is_correct = judge_equivalence(model_answer, ground_truth)

        status = "CORRECT" if is_correct else "wrong"
        print(f"  [{i+1}] {rec['math500_id']} {key} | model={model_answer!r} gt={ground_truth!r} → {status}")

        if is_correct:
            results[key]["correct"] += 1
            results[key]["rescored"] += 1

    return results


def print_table(results: dict, label: str):
    print(f"\n=== {label} ===")
    order = [
        ("contaminated", "original"),
        ("contaminated", "number_swap"),
        ("contaminated", "surface_noise"),
        ("clean", "original"),
        ("clean", "number_swap"),
        ("clean", "surface_noise"),
    ]
    for key in order:
        if key in results:
            r = results[key]
            rescored = r["rescored"]
            tag = f"  (+{rescored} rescored)" if rescored else ""
            print(f"  {key[0]:15} / {key[1]:15}: {r['correct']}/{r['total']}{tag}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results/did_traces_openthoughts.jsonl"
    print(f"Re-scoring: {path}")
    results = rescore_file(path)
    print_table(results, path)
