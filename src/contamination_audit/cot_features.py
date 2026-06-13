"""Chain-of-thought feature extraction (paper §7.3 / Table 7 / Figure 2).

Five per-trace features (linearised from cell 22 of ``downloads/analysis.ipynb``):

  - ``word_count``     — number of whitespace-delimited tokens in the trace
  - ``uncertainty``    — regex hits indicating revision / surprise ("wait", "no", ...)
  - ``hedging``        — regex hits indicating tentativeness ("I think", "perhaps", ...)
  - ``self_correction``— regex hits for explicit recalculation moves
  - ``math_density``   — fraction of tokens containing digits / operators / LaTeX cmds

Reported with Mann-Whitney U tests + Cohen's d per metric, clean vs contaminated,
on the **original** perturbation only (so any signal cannot be attributed to the
perturbation itself).
"""

from __future__ import annotations

import re
import string
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


UNCERTAINTY_PATTERNS = [
    r"\bno wait\b", r"\bwait,?\s+no\b", r"\bhold on\b",
    r"\blet me reconsider\b", r"\blet me re-?think\b", r"\blet me redo\b",
    r"\bstep back\b", r"\bi made an error\b",
    r"\bthat('s| is) (wrong|incorrect|not right)\b",
    r"\bwait,?\s+that'?s?\s+(wrong|not right|incorrect)\b",
]

HEDGING_PATTERNS = [
    r"\bi'?m not sure\b", r"\bi'?m not certain\b", r"\bnot certain\b",
    r"\bi believe\b", r"\bseems like\b", r"\bseems to be\b", r"\bprobably\b",
    r"\bperhaps\b", r"\bi think (that )?this\b",
    r"\bif i recall\b", r"\bif i remember\b",
]

SELF_CORRECT_PATTERNS = [
    r"\bactually,?\s+(no|wait|let me)\b", r"\bno,?\s+wait\b",
    r"\blet me recalculate\b", r"\blet me re-?do\b", r"\bi was wrong\b",
    r"\bi (made a |have a )?mistake\b", r"\bthat was (a )?mistake\b",
    r"\blet me correct\b", r"\bi need to correct\b",
    r"\bmy (previous |earlier )?calculation was (wrong|incorrect|off)\b",
]

MATH_TOKEN_PATTERN = re.compile(r"\d+\.?\d*|[+\-*/=^√∫∑]|\\[a-zA-Z]+")

# Strip ChatML/Tulu/OpenThoughts thought tags before tokenizing.
TAG_STRIP = re.compile(r"<\|.*?\|>")


def _count(text: str, patterns: list[str]) -> int:
    return sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)


def extract_features(trace: str | None) -> dict | None:
    """Extract the five CoT features from a single trace. Returns ``None`` for empty traces."""
    if not trace:
        return None
    text = TAG_STRIP.sub("", trace).strip()
    if not text:
        return None
    words = text.split()
    clean_words = [
        w.strip(string.punctuation).lower()
        for w in words
        if w.strip(string.punctuation)
    ]
    math_tokens = len(MATH_TOKEN_PATTERN.findall(text))
    word_count = len(words)
    return {
        "word_count": word_count,
        "uncertainty": _count(text, UNCERTAINTY_PATTERNS),
        "hedging": _count(text, HEDGING_PATTERNS),
        "self_correction": _count(text, SELF_CORRECT_PATTERNS),
        "math_density": (math_tokens / word_count) if word_count else 0.0,
        "type_token_ratio": (len(set(clean_words)) / len(clean_words)) if clean_words else 0.0,
    }


def features_dataframe(records: Iterable[dict]) -> pd.DataFrame:
    """Return a DataFrame with (model, split, perturbation_type) plus per-trace features."""
    rows = []
    for r in records:
        feats = extract_features(r.get("full_trace"))
        if feats is None:
            continue
        rows.append({
            "model": r.get("model") or r.get("dataset"),
            "split": r.get("split"),
            "perturbation_type": r.get("perturbation_type"),
            "math500_id": r.get("math500_id"),
            "llm_correct": r.get("llm_correct"),
            **feats,
        })
    return pd.DataFrame(rows)


def mann_whitney_table(
    df: pd.DataFrame,
    metrics: tuple[str, ...] = (
        "word_count", "uncertainty", "hedging", "self_correction", "math_density",
    ),
) -> pd.DataFrame:
    """Per-model, per-metric Mann-Whitney U test + Cohen's d, clean vs contaminated on originals."""
    rows = []
    orig = df[df["perturbation_type"] == "original"]
    for model, sub in orig.groupby("model"):
        clean = sub[sub["split"] == "clean"]
        contam = sub[sub["split"] == "contaminated"]
        for metric in metrics:
            a = clean[metric].dropna().values
            b = contam[metric].dropna().values
            if a.size < 2 or b.size < 2:
                continue
            u_stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            pooled_std = float(np.std(np.concatenate([a, b]))) + 1e-9
            cohens_d = (a.mean() - b.mean()) / pooled_std
            rows.append({
                "model": model,
                "metric": metric,
                "clean_mean": round(float(a.mean()), 3),
                "contam_mean": round(float(b.mean()), 3),
                "U": float(u_stat),
                "p": round(float(p), 4),
                "cohens_d": round(float(cohens_d), 2),
                "sig": _sig_label(p),
            })
    return pd.DataFrame(rows)


def _sig_label(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "~"
    return "ns"
