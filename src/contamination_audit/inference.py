"""HuggingFace inference driver for the DiD experiment (paper §5.3, Table 3).

Linearised from ``downloads/19_did_inference (2).ipynb``. Runs each configured
model on every (math500_id, split, perturbation_type) prompt in the unified
problem set, extracts ``\\boxed{}`` answers, scores against ground truth via
LaTeX-normalised string comparison, and saves traces resumably.

The driver is *resume-safe*: a prompt is considered complete iff at least
``n_samples`` records exist on disk. Interrupted runs can re-enter the same
output file; partial / corrupt lines are healed by ``repair_traces``.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceConfig:
    model_id: str
    name: str                        # short alias used in trace files (e.g. "openthoughts")
    n_samples: int = 1
    temperature: float = 0.8
    max_new_tokens: int = 4096
    use_4bit: bool = False           # set True for low-VRAM GPUs


# ── answer extraction & normalisation ─────────────────────────────────────────

_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")


def extract_boxed_answer(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def normalize_answer(ans: str | None) -> str | None:
    """Lightweight LaTeX → ASCII normalisation matching the notebook."""
    if ans is None:
        return None
    ans = ans.strip()
    ans = re.sub(r"\$", "", ans).strip()
    ans = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", ans)
    ans = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", ans)
    ans = re.sub(r"\\sqrt\s+(\w)", r"sqrt(\1)", ans)
    ans = re.sub(r"\\left|\\right|\\,|\\;|\\!|\\:", "", ans)
    ans = re.sub(r"\\cdot", "*", ans)
    ans = re.sub(r"\\times", "*", ans)
    ans = re.sub(r"\\pi", "pi", ans)
    ans = re.sub(r"\\infty", "inf", ans)
    ans = re.sub(r"\\cup", "U", ans)
    ans = re.sub(r"\\circ", "°", ans)
    ans = re.sub(r"\s+", " ", ans).strip()
    return ans.lower()


# ── model lifecycle ──────────────────────────────────────────────────────────


def load_model(model_id: str, *, use_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
    model.eval()
    return tokenizer, model


def free_memory() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_samples(tokenizer, model, problem_text: str, *,
                     n_samples: int, temperature: float, max_new_tokens: int) -> list[str]:
    import torch
    messages = [{"role": "user", "content": problem_text}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    traces: list[str] = []
    for _ in range(n_samples):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][input_len:]
        traces.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return traces


# ── trace-file health check ──────────────────────────────────────────────────


def repair_traces(path: Path, n_samples: int) -> dict[str, int]:
    """Drop corrupt / partial records from a trace file. Atomic rewrite, idempotent."""
    if not path.exists():
        return {"complete": 0, "partial_dropped": 0, "corrupt_lines": 0}

    by_prompt: dict[tuple, dict[int, dict]] = defaultdict(dict)
    n_corrupt = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_corrupt += 1
                continue
            key = (rec.get("math500_id"), rec.get("split"), rec.get("perturbation_type"))
            by_prompt[key][rec.get("sample_index", 0)] = rec

    complete = {k: v for k, v in by_prompt.items() if len(v) >= n_samples}
    partial = {k: v for k, v in by_prompt.items() if 0 < len(v) < n_samples}

    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for samples in complete.values():
            for si in sorted(samples.keys())[:n_samples]:
                f.write(json.dumps(samples[si]) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

    return {"complete": len(complete), "partial_dropped": len(partial), "corrupt_lines": n_corrupt}


# ── main run loop ────────────────────────────────────────────────────────────


def run(config: InferenceConfig, problems: Iterable[dict], out_path: Path) -> None:
    """Run inference over an iterable of problem dicts, appending to ``out_path``.

    Each problem dict must include ``math500_id``, ``split``, ``perturbation_type``,
    ``problem``, and ``answer``. Optional fields are forwarded to the trace.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    problems = list(problems)

    repair_traces(out_path, config.n_samples)

    counts: dict[tuple, int] = defaultdict(int)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (rec.get("math500_id"), rec.get("split"), rec.get("perturbation_type"))
                counts[key] += 1
    completed = {k for k, c in counts.items() if c >= config.n_samples}
    remaining = [
        p for p in problems
        if (p["math500_id"], p["split"], p["perturbation_type"]) not in completed
    ]
    if completed:
        _log.info("[%s] resuming: %d prompts already complete", config.name, len(completed))
    if not remaining:
        _log.info("[%s] nothing to do", config.name)
        return

    _log.info("[%s] loading %s (4bit=%s)", config.name, config.model_id, config.use_4bit)
    tokenizer, model = load_model(config.model_id, use_4bit=config.use_4bit)
    _log.info("[%s] running %d prompts × %d sample(s)",
              config.name, len(remaining), config.n_samples)

    try:
        with open(out_path, "a", encoding="utf-8") as out_f:
            for prob in remaining:
                gt_norm = normalize_answer(prob.get("answer")) if prob.get("answer") else None
                traces = generate_samples(
                    tokenizer, model, prob["problem"],
                    n_samples=config.n_samples,
                    temperature=config.temperature,
                    max_new_tokens=config.max_new_tokens,
                )
                for i, trace in enumerate(traces):
                    extracted = normalize_answer(extract_boxed_answer(trace))
                    if gt_norm is None:
                        correct = None
                    else:
                        correct = (extracted == gt_norm) if extracted is not None else False
                    record = {
                        "math500_id": prob["math500_id"],
                        "model": config.name,
                        "split": prob["split"],
                        "perturbation_type": prob["perturbation_type"],
                        "subject": prob.get("subject"),
                        "level": prob.get("level"),
                        "answer_confidence": prob.get("answer_confidence", ""),
                        "temperature": config.temperature,
                        "sample_index": i,
                        "full_trace": trace,
                        "final_answer": extracted,
                        "ground_truth": prob["answer"],
                        "correct": correct,
                    }
                    out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                os.fsync(out_f.fileno())
    finally:
        del model, tokenizer
        free_memory()
    _log.info("[%s] done → %s", config.name, out_path)
