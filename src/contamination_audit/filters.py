"""Faithful reimplementations of published decontamination filters.

Each entry records where its parameters come from, because that provenance is the
point: two filters below are pinned by their project's source code, one is pinned
only by its paper's prose, and they do not agree.

All filters answer the same question: which benchmark items does this filter flag
as contaminated by this corpus? They return sets of benchmark indices, so their
outputs are directly comparable.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

_SPLIT_ALNUM = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------- tokenizers

def tok_s1(text: str) -> list[str]:
    """simplescaling/s1 data/decontaminate_util.py: normalize_string + word_ngrams."""
    return " ".join(text.lower().strip().split()).split()


def tok_whitespace(text: str) -> list[str]:
    return text.split()


def tok_alnum(text: str) -> list[str]:
    return [t for t in _SPLIT_ALNUM.split(text.lower()) if t]


def _hf_tokenizer(name: str) -> Callable[[str], list[int]]:
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(name)
    return lambda text: tk.encode(text, add_special_tokens=False)


def _tiktoken(encoding: str) -> Callable[[str], list[int]]:
    import tiktoken
    enc = tiktoken.get_encoding(encoding)
    return lambda text: enc.encode(text)


# ---------------------------------------------------------------- primitives

def ngrams(tokens: Iterable, n: int) -> set[tuple]:
    toks = list(tokens)
    if len(toks) < n:
        return set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def flag_by_ngram(corpus: list[str], bench: list[str], n: int,
                  tokenize: Callable) -> set[int]:
    """Any shared n-gram flags the benchmark item. Mirrors s1's lookup design:
    index the benchmark once, then stream the corpus."""
    lookup: dict[tuple, set[int]] = {}
    for j, text in enumerate(bench):
        for g in ngrams(tokenize(text), n):
            lookup.setdefault(g, set()).add(j)

    flagged: set[int] = set()
    for text in corpus:
        if not text.strip():
            continue
        for g in ngrams(tokenize(text), n):
            hit = lookup.get(g)
            if hit:
                flagged |= hit
    return flagged


def flag_by_fuzz(corpus: list[str], bench: list[str], threshold: float) -> set[int]:
    """OpenThoughts-114K: process.extract(scorer=fuzz.ratio, score_cutoff=...)."""
    from rapidfuzz import fuzz, process
    mat = process.cdist(corpus, bench, scorer=fuzz.ratio,
                        score_cutoff=threshold, workers=-1)
    return {j for i in range(mat.shape[0]) for j in range(mat.shape[1])
            if mat[i][j] >= threshold}


def flag_by_coverage(corpus: list[str], bench: list[str], n: int,
                     coverage: float, tokenize: Callable) -> set[int]:
    """Tulu 3: fraction of benchmark tokens covered by shared n-grams."""
    bench_toks = [tokenize(t) for t in bench]
    bench_grams = [ngrams(t, n) for t in bench_toks]
    covered: list[set[int]] = [set() for _ in bench]

    for text in corpus:
        if not text.strip():
            continue
        cg = ngrams(tokenize(text), n)
        if not cg:
            continue
        for j, bg in enumerate(bench_grams):
            shared = bg & cg
            if not shared:
                continue
            toks = bench_toks[j]
            for i in range(len(toks) - n + 1):
                if tuple(toks[i:i + n]) in shared:
                    covered[j].update(range(i, i + n))

    return {j for j in range(len(bench))
            if bench_toks[j] and len(covered[j]) / len(bench_toks[j]) > coverage}


# ---------------------------------------------------------------- the registry

@dataclass(frozen=True)
class DecontaminationFilter:
    name: str
    project: str
    provenance: str          # "code" if parameters are defaults in released source
    description: str
    run: Callable[[list[str], list[str]], set[int]]
    faithful: bool = True    # False = partial reimplementation, NOT comparable


def registry(include_partial: bool = False) -> list[DecontaminationFilter]:
    """Faithful filters only by default.

    A partial reimplementation must never sit in a comparison table beside complete
    ones: its number would be read as that project's behaviour. Opting in is
    deliberate and its output is for diagnosis, not for reporting.
    """
    all_filters = [
        DecontaminationFilter(
            "s1-8gram-word", "s1", "code",
            "any shared 8-gram over lowercased whitespace tokens",
            lambda c, b: flag_by_ngram(c, b, 8, tok_s1),
        ),
        DecontaminationFilter(
            "ot3-13gram-qwen", "OpenThoughts3", "code",
            "any shared 13-gram over Qwen/Qwen2-7B-Instruct tokens",
            lambda c, b: flag_by_ngram(c, b, 13, _hf_tokenizer("Qwen/Qwen2-7B-Instruct")),
        ),
        DecontaminationFilter(
            "ot114k-fuzz95", "OpenThoughts-114K", "code",
            "rapidfuzz fuzz.ratio >= 95 on raw strings",
            lambda c, b: flag_by_fuzz(c, b, 95.0),
        ),
        DecontaminationFilter(
            "PARTIAL-cl100k-5gram-candidates-only", "NOT decon", "code",
            "decon's candidate-generation stage ONLY: any shared 5-gram over cl100k "
            "tokens. decon then applies cluster expansion and a weighted score gated "
            "at contamination_score_threshold=0.8 (Olmo 3 Appendix A.5), which is not "
            "implemented here. MEASURED: on 100k Dolci questions this flags 199/200 "
            "OlymMATH items; the real decon v0.3.0 binary on the same input flags 0. "
            "The approximation inverts the result. Kept only as a worked example that "
            "a two-stage filter cannot be approximated by its first stage. To measure "
            "decon, build github.com/allenai/decon and run `decon detect`.",
            lambda c, b: flag_by_ngram(c, b, 5, _tiktoken("cl100k_base")),
            faithful=False,
        ),
        DecontaminationFilter(
            "tulu3-8gram-50pct", "Tulu 3", "prose",
            "8-gram, >50% benchmark-token coverage -- values from the paper only; "
            "the released code declares no defaults for either",
            lambda c, b: flag_by_coverage(c, b, 8, 0.5, tok_whitespace),
        ),
    ]
    return all_filters if include_partial else [f for f in all_filters if f.faithful]
