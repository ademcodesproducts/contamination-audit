"""Matching rules: how a filter decides that a benchmark item is contaminated.

Every rule answers the same question -- which items of this benchmark does
this corpus contaminate? -- and returns item indices, so any two rules can be
compared directly. Rules combine with `|`:

    SharedNgram(13, QWEN2) | FuzzyRatio(75)

flags an item when either rule would.

The n-gram rules share one design, taken from s1's decontamination code: index
the benchmark once, then stream the corpus past the index. Scoring a document
costs only the n-grams it actually shares with the benchmark.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from .tokenizers import S1_WORDS, Tokenizer

NGram = tuple[Hashable, ...]


def ngrams(tokens: Sequence[Hashable], n: int) -> set[NGram]:
    """The distinct n-grams of a token sequence, words and token ids alike."""
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class MatchingRule(ABC):
    """A way of deciding which benchmark items a corpus contaminates.

    Every rule decides each benchmark item on its own: whether item i is
    flagged never depends on which other items are being checked.
    """

    @abstractmethod
    def flag(self, corpus: Sequence[str], benchmark: Sequence[str]) -> set[int]:
        """Indices of the benchmark items this rule finds in the corpus."""

    def flag_groups(self, corpus: Sequence[str], *groups: Sequence[str]) -> list[set[int]]:
        """Flag several benchmarks with a single pass over the corpus.

        Because items are decided independently, flagging the groups together
        and splitting the answer is the same as flagging each group alone, and
        it tokenizes the corpus once rather than once per group.
        """
        flagged = self.flag(corpus, [text for group in groups for text in group])
        per_group, start = [], 0
        for group in groups:
            end = start + len(group)
            per_group.append({i - start for i in flagged if start <= i < end})
            start = end
        return per_group

    def __or__(self, other: MatchingRule) -> AnyOf:
        return AnyOf((self, other))


@dataclass(frozen=True)
class SharedNgram(MatchingRule):
    """Flag an item if any one of its n-grams appears in any training document."""

    n: int
    tokenizer: Tokenizer

    def flag(self, corpus, benchmark):
        items_with: dict[NGram, set[int]] = {}
        for item, text in enumerate(benchmark):
            for gram in ngrams(self.tokenizer(text), self.n):
                items_with.setdefault(gram, set()).add(item)

        flagged: set[int] = set()
        for document in corpus:
            if not document.strip():
                continue
            for gram in ngrams(self.tokenizer(document), self.n):
                hit = items_with.get(gram)
                if hit:
                    flagged |= hit
        return flagged


def best_fuzzy_scores(corpus: Sequence[str], benchmark: Sequence[str], floor: float):
    """Each item's best `fuzz.ratio` against any document; scores below `floor` read 0.

    rapidfuzz computes the corpus-by-benchmark matrix in C, and numpy reduces it
    over the corpus axis. Walking the matrix in Python would take tens of
    millions of interpreter steps to find a handful of hits.
    """
    import numpy as np
    from rapidfuzz import fuzz, process

    scores = process.cdist(corpus, benchmark, scorer=fuzz.ratio, score_cutoff=floor, workers=-1)
    return np.asarray(scores).max(axis=0)


@dataclass(frozen=True)
class FuzzyRatio(MatchingRule):
    """Flag an item if some document's whole-string similarity reaches the threshold.

    Similarity is rapidfuzz's `fuzz.ratio` on raw strings, as OpenThoughts uses.
    """

    threshold: float

    def flag(self, corpus, benchmark):
        if not corpus:
            return set()
        best = best_fuzzy_scores(corpus, benchmark, floor=self.threshold)
        return set((best >= self.threshold).nonzero()[0].tolist())


@dataclass(frozen=True)
class TokenCoverage(MatchingRule):
    """Flag an item if one document's shared n-grams cover more than `threshold` of it.

    Coverage is the fraction of the item's tokens that fall inside some n-gram
    the document shares with it. It is scored per document and maximised over
    documents, as open-instruct does (decontamination/search.py keys matches by
    doc_id and reads the best one). Pooling across documents would let fragments
    of unrelated documents clear the threshold together.
    """

    n: int
    threshold: float
    tokenizer: Tokenizer

    def flag(self, corpus, benchmark):
        item_tokens = [self.tokenizer(text) for text in benchmark]

        # Each n-gram maps to the (item, first position) pairs it starts, so a
        # shared n-gram turns straight into a span of covered positions.
        spans_of: dict[NGram, set[tuple[int, int]]] = {}
        for item, tokens in enumerate(item_tokens):
            for start in range(len(tokens) - self.n + 1):
                gram = tuple(tokens[start : start + self.n])
                spans_of.setdefault(gram, set()).add((item, start))

        best_covered: dict[int, int] = {}
        for document in corpus:
            if not document.strip():
                continue
            covered: dict[int, set[int]] = {}
            for gram in ngrams(self.tokenizer(document), self.n):
                for item, start in spans_of.get(gram, ()):
                    covered.setdefault(item, set()).update(range(start, start + self.n))
            for item, positions in covered.items():
                best_covered[item] = max(best_covered.get(item, 0), len(positions))

        # Strictly greater: Tulu 3's paper says "greater than 50%".
        return {
            item
            for item, covered in best_covered.items()
            if item_tokens[item] and covered / len(item_tokens[item]) > self.threshold
        }


@dataclass(frozen=True)
class PhraseAgreement(MatchingRule):
    """Flag an item only if one document contains enough of its sampled phrases.

    Samples `phrases` n-grams spread evenly through each item, in the order they
    occur, and requires at least `required_fraction` of them in a single
    document. Demanding agreement across the whole item is meant to suppress the
    match on one conventional phrase that causes most false positives.
    """

    required_fraction: float = 0.5
    n: int = 8
    phrases: int = 20
    tokenizer: Tokenizer = S1_WORDS

    def sample_phrases(self, text: str) -> list[NGram]:
        """Evenly spaced windows, in the order they occur in the item.

        Sampling from the set `ngrams` returns would space the phrases through
        hash order rather than through the problem, which is the one property
        this rule depends on.
        """
        tokens = self.tokenizer(text)
        windows = [tuple(tokens[p : p + self.n]) for p in range(len(tokens) - self.n + 1)]
        if len(windows) <= self.phrases:
            return windows
        step = len(windows) / self.phrases
        return [windows[int(i * step)] for i in range(self.phrases)]

    def flag(self, corpus, benchmark):
        phrase_slots: dict[NGram, set[tuple[int, int]]] = {}
        sampled_per_item: list[int] = []
        for item, text in enumerate(benchmark):
            sampled = self.sample_phrases(text)
            sampled_per_item.append(len(sampled))
            for slot, phrase in enumerate(sampled):
                phrase_slots.setdefault(phrase, set()).add((item, slot))

        # Agreement is counted within one document, then maximised over
        # documents, so unrelated documents cannot each contribute a phrase.
        best_found: dict[int, int] = {}
        for document in corpus:
            if not document.strip():
                continue
            found: dict[int, set[int]] = {}
            for gram in ngrams(self.tokenizer(document), self.n):
                for item, slot in phrase_slots.get(gram, ()):
                    found.setdefault(item, set()).add(slot)
            for item, slots in found.items():
                best_found[item] = max(best_found.get(item, 0), len(slots))

        return {
            item
            for item, found in best_found.items()
            if sampled_per_item[item] and found / sampled_per_item[item] >= self.required_fraction
        }


@dataclass(frozen=True)
class AnyOf(MatchingRule):
    """Flag an item if any of the component rules does."""

    rules: tuple[MatchingRule, ...]

    def flag(self, corpus, benchmark):
        return set().union(*(rule.flag(corpus, benchmark) for rule in self.rules))

    def __or__(self, other: MatchingRule) -> AnyOf:
        return AnyOf((*self.rules, other))
