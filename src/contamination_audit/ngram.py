"""N-gram contamination filter (Stage 1 of the detection pipeline).

Replicates each audited project's exact n-gram filter using the Qwen2-7B tokenizer:
  s1 / OpenThoughts — flag any pair sharing ≥1 n-gram (8 / 13 respectively).
  Tülu 3            — flag any pair whose test-side token coverage exceeds 50%.

A pair flagged here is a *Type 1 contamination*: it survived a filter the project
claims to apply. The release dataset should not contain such items.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from tqdm import tqdm

from .config import NgramConfig

_log = logging.getLogger(__name__)


@dataclass
class NgramHit:
    project: str
    math500_id: str
    math500_problem: str
    math500_answer: str
    math500_subject: str
    math500_level: int
    train_id: str
    train_problem: str
    n_shared_ngrams: int
    token_coverage: float
    ngram_n: int
    threshold_mode: str

    def to_dict(self) -> dict:
        return {
            "project": self.project,
            "contamination_type": "c_lex",
            "math500_id": self.math500_id,
            "math500_problem": self.math500_problem,
            "math500_answer": self.math500_answer,
            "math500_subject": self.math500_subject,
            "math500_level": self.math500_level,
            "train_id": self.train_id,
            "train_problem": self.train_problem,
            "n_shared_ngrams": self.n_shared_ngrams,
            "token_coverage": round(self.token_coverage, 4),
            "ngram_n": self.ngram_n,
            "threshold_mode": self.threshold_mode,
            "filter_should_have_caught": True,
        }


def extract_ngrams(tokens: list[int], n: int) -> set[tuple[int, ...]]:
    """Return the set of unique n-gram tuples in a token sequence."""
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def token_coverage(test_tokens: list[int], shared: set[tuple[int, ...]], n: int) -> float:
    """Fraction of test tokens covered by any shared n-gram (Tülu 3 metric)."""
    if not test_tokens:
        return 0.0
    covered: set[int] = set()
    for ngram in shared:
        for i in range(len(test_tokens) - n + 1):
            if tuple(test_tokens[i : i + n]) == ngram:
                covered.update(range(i, i + n))
    return len(covered) / len(test_tokens)


def _load_tokenizer(name: str):
    from transformers import AutoTokenizer  # local import — heavy
    return AutoTokenizer.from_pretrained(name)


def find_hits(
    project: str,
    train_items: list[dict],
    math500_items: list[dict],
    config: NgramConfig,
) -> list[dict]:
    """Run the n-gram filter and return hit records ready for jsonl dump.

    Hits are deduplicated per (train_id, math500_id) pair by the inner loop's
    set-intersection semantics; downstream stages may further dedupe by
    math500_id when assembling the final C_lex set.
    """
    tokenizer = _load_tokenizer(config.tokenizer)
    n = config.n

    _log.info(
        "n-gram audit: project=%s n=%d threshold=%s train=%d math500=%d",
        project, n, config.threshold_mode, len(train_items), len(math500_items),
    )

    math500_cache: list[tuple[dict, set[tuple[int, ...]], list[int]]] = []
    for item in math500_items:
        text = item.get("problem", "")
        tokens = tokenizer.encode(text, add_special_tokens=False) if text.strip() else []
        math500_cache.append((item, extract_ngrams(tokens, n), tokens))

    hits: list[dict] = []
    for train_item in tqdm(train_items, desc=f"{project} n-gram"):
        train_text = train_item.get("problem", "")
        if not train_text.strip():
            continue
        train_tokens = tokenizer.encode(train_text, add_special_tokens=False)
        train_ngrams = extract_ngrams(train_tokens, n)
        if not train_ngrams:
            continue

        for math_item, math_ngrams, math_tokens in math500_cache:
            shared = train_ngrams & math_ngrams
            if not shared:
                continue

            if config.threshold_mode == "any":
                coverage = 0.0
                is_hit = True
            else:
                coverage = token_coverage(math_tokens, shared, n)
                is_hit = coverage > config.coverage

            if not is_hit:
                continue

            hits.append(
                NgramHit(
                    project=project,
                    math500_id=math_item["math500_id"],
                    math500_problem=math_item.get("problem", ""),
                    math500_answer=math_item.get("answer", ""),
                    math500_subject=math_item.get("subject", ""),
                    math500_level=math_item.get("level", -1),
                    train_id=train_item["train_id"],
                    train_problem=train_text,
                    n_shared_ngrams=len(shared),
                    token_coverage=coverage,
                    ngram_n=n,
                    threshold_mode=config.threshold_mode,
                ).to_dict()
            )

    unique_math = len({h["math500_id"] for h in hits})
    _log.info("  %d hits  (%d unique MATH-500 items)", len(hits), unique_math)
    return hits


def dedup_by_math500(hits: Iterable[dict], min_shared: int) -> list[dict]:
    """Filter by minimum shared-n-gram count, then keep the strongest hit per math500_id.

    Used by ``05_validate_and_report.py`` to produce the conservative C_lex
    estimate (default ``min_shared=5`` per the paper).
    """
    filtered = [h for h in hits if h["n_shared_ngrams"] >= min_shared]
    seen: set[str] = set()
    unique: list[dict] = []
    for item in sorted(filtered, key=lambda h: -h["n_shared_ngrams"]):
        if item["math500_id"] not in seen:
            seen.add(item["math500_id"])
            unique.append(item)
    return unique
