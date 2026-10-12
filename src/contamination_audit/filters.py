"""Faithful reimplementations of published decontamination filters.

Each entry records where its parameters come from, because that provenance is the
point: two filters below are pinned by their project's source code, one is pinned
only by its paper's prose, and they do not agree.

All filters answer the same question: which benchmark items does this filter flag
as contaminated by this corpus? They return sets of benchmark indices, so their
outputs are directly comparable.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

# A token is a word or a model token id; the filters never need to know which.
Token = TypeVar("Token", str, int)

_SPLIT_ALNUM = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------- tokenizers


def tok_s1(text: str) -> list[str]:
    """simplescaling/s1 data/decontaminate_util.py: normalize_string + word_ngrams."""
    return " ".join(text.lower().strip().split()).split()


def tok_whitespace(text: str) -> list[str]:
    return text.split()


def tok_alnum(text: str) -> list[str]:
    return [t for t in _SPLIT_ALNUM.split(text.lower()) if t]


# Cached: the registry's filters look their tokenizer up on every call, and
# loading a HuggingFace tokenizer takes about a second.
@functools.cache
def _hf_tokenizer(name: str) -> Callable[[str], list[int]]:
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(name)
    return lambda text: tk.encode(text, add_special_tokens=False)


@functools.cache
def _tiktoken(encoding: str) -> Callable[[str], list[int]]:
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    return lambda text: enc.encode(text)


@functools.cache
def _spacy_tokenizer():
    import spacy

    # Only the tokenizer: open-instruct calls the loaded pipeline and reads
    # token.text, which the tagger, parser and NER do not affect. Verified
    # identical on 261 corpus and benchmark strings, and about 150x faster.
    return spacy.load("en_core_web_lg").tokenizer


def tok_spacy(text: str) -> list[str]:
    """Tulu 3's tokenizer: allenai/open-instruct decontamination/search.py:16,87."""
    return [token.text for token in _spacy_tokenizer()(text)]


def tok_qwen(text: str) -> list[int]:
    """OpenThoughts3's tokenizer. Loaded on first call, not at import."""
    return _hf_tokenizer("Qwen/Qwen2-7B-Instruct")(text)


def tok_cl100k(text: str) -> list[int]:
    """decon's tokenizer. Loaded on first call, not at import."""
    return _tiktoken("cl100k_base")(text)


# ---------------------------------------------------------------- primitives


def ngrams(tokens: Sequence[Token], n: int) -> set[tuple[Token, ...]]:
    """Distinct n-grams, over word strings or over model token ids alike.

    Nothing downstream cares which, so the filters differ only in the tokenizer
    they are handed. That is the axis the paper measures.
    """
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def flag_by_ngram(
    corpus: list[str], bench: list[str], n: int, tokenize: Callable
) -> set[int]:
    """Any shared n-gram flags the benchmark item. Mirrors s1's lookup design:
    index the benchmark once, then stream the corpus."""
    lookup: dict[tuple, set[int]] = {}
    for item, text in enumerate(bench):
        for gram in ngrams(tokenize(text), n):
            lookup.setdefault(gram, set()).add(item)

    flagged: set[int] = set()
    for text in corpus:
        if not text.strip():
            continue
        for gram in ngrams(tokenize(text), n):
            hit = lookup.get(gram)
            if hit:
                flagged |= hit
    return flagged


def flag_by_fuzz(corpus: list[str], bench: list[str], threshold: float) -> set[int]:
    """OpenThoughts-114K: process.extract(scorer=fuzz.ratio, score_cutoff=...).

    cdist does the work in C; reduce over the corpus axis in numpy rather than
    walking the matrix cell by cell, which at 100k x 800 is 80M interpreter steps
    to find a handful of hits.
    """
    import numpy as np
    from rapidfuzz import fuzz, process

    scores = process.cdist(
        corpus, bench, scorer=fuzz.ratio, score_cutoff=threshold, workers=-1
    )
    best_per_item = np.asarray(scores).max(axis=0)
    return set(np.flatnonzero(best_per_item >= threshold).tolist())


def flag_by_coverage(
    corpus: list[str], bench: list[str], n: int, coverage: float, tokenize: Callable
) -> set[int]:
    """Tulu 3: fraction of a benchmark item's tokens covered by shared n-grams.

    Indexed the same way as `flag_by_ngram`. The obvious version compares every
    corpus document against every benchmark item and then re-scans the item to
    find where the match landed; that is quadratic in the wrong places and took
    roughly 400s on a 334k-document corpus. Precomputing each n-gram's token span
    once makes a corpus document cost only the spans it actually hits.
    """
    bench_tokens = [tokenize(t) for t in bench]

    # n-gram -> {(item, first token index)}, so a hit maps straight to a span.
    lookup: dict[tuple, set[tuple[int, int]]] = {}
    for item, tokens in enumerate(bench_tokens):
        for position in range(len(tokens) - n + 1):
            gram = tuple(tokens[position : position + n])
            lookup.setdefault(gram, set()).add((item, position))

    # Coverage is scored per training document, then maximised over documents:
    # open-instruct keys its matches by doc_id and reads the best single
    # document's score (decontamination/search.py:102,133). Pooling positions
    # across the corpus instead lets fragments from unrelated documents clear
    # the threshold together, flagging items no one document covers.
    best: dict[int, int] = {}
    for text in corpus:
        if not text.strip():
            continue
        covered: dict[int, set[int]] = {}
        for gram in ngrams(tokenize(text), n):
            for item, position in lookup.get(gram, ()):
                covered.setdefault(item, set()).update(range(position, position + n))
        for item, marked in covered.items():
            if len(marked) > best.get(item, 0):
                best[item] = len(marked)

    return {
        item
        for item, marked in best.items()
        if bench_tokens[item] and marked / len(bench_tokens[item]) > coverage
    }


def flag_by_sampled_phrases(
    corpus: list[str],
    bench: list[str],
    n: int = 8,
    n_phrases: int = 20,
    frac: float = 0.5,
    tokenize: Callable = tok_s1,
    exhaustive: bool = False,
) -> set[int]:
    """Rufus-Air style: require a fraction of an item's phrases to be present.

    Rather than firing on any single shared n-gram, this samples roughly
    `n_phrases` evenly spaced n-grams per benchmark item and flags the item only
    when at least `frac` of them appear somewhere in the corpus. Requiring
    agreement across spans spread through the item is meant to suppress the
    single-stock-phrase match that drives the false positives in this paper.

    `exhaustive=True` uses every n-gram instead of a sample, the variant reported
    for mathematics.
    """
    # phrase -> {(benchmark item, which of that item's phrases this is)}
    lookup: dict[tuple, set[tuple[int, int]]] = {}
    n_sampled: list[int] = []
    for item, text in enumerate(bench):
        # Windows in the order they occur in the item. `ngrams` returns a set,
        # so sampling from it spaces the phrases through a hash or alphabetical
        # ordering rather than through the problem, which is the one property
        # this filter depends on.
        tokens = tokenize(text)
        grams = [tuple(tokens[p : p + n]) for p in range(len(tokens) - n + 1)]
        if not grams:
            n_sampled.append(0)
            continue
        if exhaustive or len(grams) <= n_phrases:
            picked = grams
        else:
            # Evenly spaced rather than contiguous: the phrases have to be
            # spread through the item for their agreement to mean anything.
            step = len(grams) / n_phrases
            picked = [grams[int(i * step)] for i in range(n_phrases)]
        n_sampled.append(len(picked))
        for position, gram in enumerate(picked):
            lookup.setdefault(gram, set()).add((item, position))

    # Agreement is counted within one document, then maximised over documents.
    # Pooling across the corpus would let unrelated documents each contribute a
    # phrase and jointly clear `frac`, which is the single-stock-phrase match
    # this filter exists to suppress.
    best: dict[int, int] = {}
    for text in corpus:
        if not text.strip():
            continue
        found: dict[int, set[int]] = {}
        for gram in ngrams(tokenize(text), n):
            for item, position in lookup.get(gram, ()):
                found.setdefault(item, set()).add(position)
        for item, positions in found.items():
            if len(positions) > best.get(item, 0):
                best[item] = len(positions)

    return {
        item
        for item, hits in best.items()
        if n_sampled[item] and hits / n_sampled[item] >= frac
    }


# ---------------------------------------------------------------- the registry


@dataclass(frozen=True)
class DecontaminationFilter:
    name: str
    project: str
    provenance: str  # "code" if parameters are defaults in released source
    description: str
    run: Callable[[list[str], list[str]], set[int]]
    faithful: bool = True  # False = partial reimplementation, NOT comparable


def registry(include_partial: bool = False) -> list[DecontaminationFilter]:
    """Faithful filters only by default.

    A partial reimplementation must never sit in a comparison table beside complete
    ones: its number would be read as that project's behaviour. Opting in is
    deliberate and its output is for diagnosis, not for reporting.
    """
    all_filters = [
        DecontaminationFilter(
            "s1-8gram-word",
            "s1",
            "code",
            "any shared 8-gram over lowercased whitespace tokens",
            lambda c, b: flag_by_ngram(c, b, 8, tok_s1),
        ),
        DecontaminationFilter(
            "ot3-13gram-qwen-or-fuzz75",
            "OpenThoughts3",
            "code",
            "any shared 13-gram over Qwen/Qwen2-7B-Instruct tokens OR "
            "rapidfuzz fuzz.ratio >= 75. Both branches, as the paper states "
            "(Appendix F, 'marked as contaminated by either of our methods'). "
            "The released math config sets the fuzzy cutoff to 80, which adds "
            "no further items on this corpus.",
            lambda c, b: flag_by_ngram(c, b, 13, tok_qwen) | flag_by_fuzz(c, b, 75.0),
        ),
        DecontaminationFilter(
            "ot114k-fuzz95",
            "OpenThoughts-114K",
            "code",
            "rapidfuzz fuzz.ratio >= 95 on raw strings",
            lambda c, b: flag_by_fuzz(c, b, 95.0),
        ),
        DecontaminationFilter(
            "PARTIAL-cl100k-5gram-candidates-only",
            "NOT decon",
            "code",
            "decon's candidate-generation stage ONLY: any shared 5-gram over cl100k "
            "tokens. decon then applies cluster expansion and a weighted score gated "
            "at contamination_score_threshold=0.8 (Olmo 3 Appendix A.5), which is not "
            "implemented here. MEASURED: on 100k Dolci questions this flags 199/200 "
            "OlymMATH items; the real decon v0.3.0 binary on the same input flags 0. "
            "The approximation inverts the result. Kept only as a worked example that "
            "a two-stage filter cannot be approximated by its first stage. To measure "
            "decon, build github.com/allenai/decon and run `decon detect`.",
            lambda c, b: flag_by_ngram(c, b, 5, tok_cl100k),
            faithful=False,
        ),
        DecontaminationFilter(
            "tulu3-8gram-50pct",
            "Tulu 3",
            "prose",
            "8-gram, >50% benchmark-token coverage, scored per training document "
            "and maximised over documents, over spaCy en_core_web_lg tokens -- "
            "the aggregation and tokenizer open-instruct uses "
            "(decontamination/search.py:16,87,102,133). n and the threshold come "
            "from the paper; the released code declares no defaults for either. "
            "One departure remains: open-instruct reaches candidate documents "
            "through Elasticsearch, capped at --search_size (default 100) hits "
            "per n-gram, where this scans the corpus exhaustively. Seeing every "
            "matching document can only raise the per-item maximum, so this is "
            "an upper bound on what the retrieval-gated original would flag.",
            lambda c, b: flag_by_coverage(c, b, 8, 0.5, tok_spacy),
        ),
    ]
    return all_filters if include_partial else [f for f in all_filters if f.faithful]
