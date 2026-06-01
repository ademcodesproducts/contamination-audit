"""Stage 8: extract \\boxed{} answers from inference traces and judge correctness.

Folds in the previous top-level llm_judge_openthoughts.py and llm_rescore.py:
  1. Try ``answer_extract.extract_boxed`` (regex).
  2. Fall back to natural-language patterns via ``answer_extract.extract_from_trace``.
  3. If still null, ask the LLM (OpenAI gpt-4o-mini by default; ``--provider anthropic``
     to use Claude Haiku 4.5 with claude-haiku-4-5-20251001).
  4. Judge equivalence to ground_truth via the same provider.

Reads:  results/traces/{model}_traces.jsonl
Writes: results/traces/{model}_traces_judged.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

import _common  # noqa: F401

from contamination_audit.answer_extract import extract_boxed, extract_from_trace
from contamination_audit.config import configure_logging
from contamination_audit.io import REPO_ROOT
from contamination_audit.judge import load_prompt

_log = logging.getLogger("score_answers")


def _make_client(provider: str):
    load_dotenv()
    if provider == "openai":
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAI(api_key=api_key)
    if provider == "anthropic":
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return anthropic.Anthropic()
    raise ValueError(f"Unknown provider: {provider}")


def _llm_extract(client, provider: str, model: str, trace: str) -> str | None:
    prompt = load_prompt("answer_extract").format(trace=trace[-3000:])
    if provider == "openai":
        resp = client.chat.completions.create(
            model=model, max_tokens=50, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()
    else:
        resp = client.messages.create(
            model=model, max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
    return None if text.upper() == "NULL" else text


def _llm_judge_equivalence(client, provider: str, model: str, gt: str, ans: str) -> bool:
    prompt = load_prompt("answer_equivalence").format(ground_truth=gt, model_answer=ans)
    if provider == "openai":
        resp = client.chat.completions.create(
            model=model, max_tokens=5, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = resp.choices[0].message.content.strip().upper()
    else:
        resp = client.messages.create(
            model=model, max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = resp.content[0].text.strip().upper()
    return verdict.startswith("YES") or "CORRECT" in verdict


def judge_record(client, provider: str, model: str, rec: dict) -> dict:
    """Augment a trace record with extracted answer + LLM correctness verdict."""
    model_answer = rec.get("final_answer")
    extracted = False
    if model_answer is None:
        # Cheap regex first.
        model_answer = extract_from_trace(rec.get("full_trace", ""))
        if model_answer is not None:
            extracted = True

    if model_answer is None:
        # Last resort: LLM extraction (used by Appendix E error-class fixes).
        try:
            model_answer = _llm_extract(client, provider, model, rec.get("full_trace", ""))
            extracted = model_answer is not None
        except Exception as e:  # noqa: BLE001
            _log.warning("extract failed for %s: %s", rec.get("math500_id"), e)

    if model_answer is None:
        return {**rec, "llm_correct": False, "llm_model_answer": None, "extracted": False}

    try:
        is_correct = _llm_judge_equivalence(client, provider, model, rec["ground_truth"], model_answer)
    except Exception as e:  # noqa: BLE001
        _log.warning("judge failed for %s: %s", rec.get("math500_id"), e)
        is_correct = False

    return {**rec, "llm_correct": is_correct, "llm_model_answer": model_answer, "extracted": extracted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", nargs="?",
                        default=str(REPO_ROOT / "results" / "traces" / "openthoughts_traces.jsonl"))
    parser.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--model", default=None,
                        help="Override the model id. Defaults: gpt-4o-mini / claude-haiku-4-5-20251001")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    configure_logging()
    model = args.model or ("gpt-4o-mini" if args.provider == "openai" else "claude-haiku-4-5-20251001")
    client = _make_client(args.provider)

    trace_path = Path(args.trace)
    with open(trace_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    _log.info("scoring %d records with %s/%s", len(records), args.provider, model)
    judged: list[dict] = [None] * len(records)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(judge_record, client, args.provider, model, rec): i
            for i, rec in enumerate(records)
        }
        for future in tqdm(as_completed(futures), total=len(records), desc="scoring"):
            i = futures[future]
            judged[i] = future.result()

    out_path = trace_path.with_name(trace_path.stem.replace("_traces", "_traces_judged") + ".jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for record in judged:
            f.write(json.dumps(record) + "\n")
    _log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
