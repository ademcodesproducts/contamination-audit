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
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from .tokenizers import S1_WORDS, Tokenizer

NGram = tuple[Hashable, ...]


def ngrams(tokens: Sequence[Hashable], n: int) -> set[NGram]:
    """The distinct n-grams of a token sequence, words and token ids alike."""
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


Matches = dict[int, set[int]]  # training document index -> benchmark items it matches


class MatchingRule(ABC):
    """A way of deciding which benchmark items a corpus contaminates.

    A rule decides, for each training document, which benchmark items it
    matches. An item is flagged when any document matches it, and a filter
    removes every document that matches something. Rules decide each benchmark
    item on its own: whether item i is flagged never depends on which other
    items are being checked.
    """

    @abstractmethod
    def match(self, corpus: Sequence[str], benchmark: Sequence[str]) -> Matches:
        """For each training document that matches anything, the items it matches."""

    def flag(self, corpus: Sequence[str], benchmark: Sequence[str]) -> set[int]:
        """Indices of the benchmark items this rule finds in the corpus."""
        return set().union(*self.match(corpus, benchmark).values())

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

    def match(self, corpus, benchmark):
        items_with: dict[NGram, set[int]] = {}
        for item, text in enumerate(benchmark):
            for gram in ngrams(self.tokenizer(text), self.n):
                items_with.setdefault(gram, set()).add(item)

        matches: Matches = {}
        for index, document in enumerate(corpus):
            if not document.strip():
                continue
            matched: set[int] = set()
            for gram in ngrams(self.tokenizer(document), self.n):
                hit = items_with.get(gram)
                if hit:
                    matched |= hit
            if matched:
                matches[index] = matched
        return matches


def best_fuzzy_scores(corpus: Sequence[str], benchmark: Sequence[str], floor: float):
    """Each item's best `fuzz.ratio` against any document; scores below `floor` read 0.

    rapidfuzz computes the corpus-by-benchmark matrix in C, and numpy reduces it
    over the corpus axis. Walking the matrix in Python would take tens of
    millions of interpreter steps to find a handful of hits.
    """
    import numpy as np

    best = np.zeros(len(benchmark))
    for _, scores in fuzzy_score_chunks(corpus, benchmark, floor):
        best = np.maximum(best, scores.max(axis=0))
    return best


# Documents scored per rapidfuzz call. The score matrix is documents by items,
# so an unchunked call on two million documents would need several gigabytes.
FUZZY_CHUNK = 50_000


def fuzzy_score_chunks(corpus: Sequence[str], benchmark: Sequence[str], floor: float):
    """The corpus-by-benchmark fuzz.ratio matrix, a block of documents at a time.

    Yields (index of the block's first document, its scores); scores below
    `floor` read 0.
    """
    import numpy as np
    from rapidfuzz import fuzz, process

    for start in range(0, len(corpus), FUZZY_CHUNK):
        block = corpus[start : start + FUZZY_CHUNK]
        # Scores stay floating point: rounding 94.6 up to 95 would pass a cutoff of 95.
        scores = process.cdist(block, benchmark, scorer=fuzz.ratio, score_cutoff=floor, workers=-1)
        yield start, np.asarray(scores)


@dataclass(frozen=True)
class FuzzyRatio(MatchingRule):
    """Flag an item if some document's whole-string similarity reaches the threshold.

    Similarity is rapidfuzz's `fuzz.ratio` on raw strings, as OpenThoughts uses.
    """

    threshold: float

    def match(self, corpus, benchmark):
        import numpy as np

        matches: Matches = {}
        for start, scores in fuzzy_score_chunks(corpus, benchmark, self.threshold):
            documents, items = np.nonzero(scores >= self.threshold)
            for document, item in zip(documents.tolist(), items.tolist(), strict=True):
                matches.setdefault(start + document, set()).add(item)
        return matches


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

    def match(self, corpus, benchmark):
        item_tokens = [self.tokenizer(text) for text in benchmark]

        # Each n-gram maps to the (item, first position) pairs it starts, so a
        # shared n-gram turns straight into a span of covered positions.
        spans_of: dict[NGram, set[tuple[int, int]]] = {}
        for item, tokens in enumerate(item_tokens):
            for start in range(len(tokens) - self.n + 1):
                gram = tuple(tokens[start : start + self.n])
                spans_of.setdefault(gram, set()).add((item, start))

        matches: Matches = {}
        for index, document in enumerate(corpus):
            if not document.strip():
                continue
            covered: dict[int, set[int]] = {}
            for gram in ngrams(self.tokenizer(document), self.n):
                for item, start in spans_of.get(gram, ()):
                    covered.setdefault(item, set()).update(range(start, start + self.n))
            # Strictly greater: Tulu 3's paper says "greater than 50%".
            matched = {
                item
                for item, positions in covered.items()
                if len(positions) / len(item_tokens[item]) > self.threshold
            }
            if matched:
                matches[index] = matched
        return matches


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

    def match(self, corpus, benchmark):
        phrase_slots: dict[NGram, set[tuple[int, int]]] = {}
        sampled_per_item: list[int] = []
        for item, text in enumerate(benchmark):
            sampled = self.sample_phrases(text)
            sampled_per_item.append(len(sampled))
            for slot, phrase in enumerate(sampled):
                phrase_slots.setdefault(phrase, set()).add((item, slot))

        # Agreement is counted within one document, so unrelated documents
        # cannot each contribute a phrase.
        matches: Matches = {}
        for index, document in enumerate(corpus):
            if not document.strip():
                continue
            found: dict[int, set[int]] = {}
            for gram in ngrams(self.tokenizer(document), self.n):
                for item, slot in phrase_slots.get(gram, ()):
                    found.setdefault(item, set()).add(slot)
            matched = {
                item
                for item, slots in found.items()
                if len(slots) / sampled_per_item[item] >= self.required_fraction
            }
            if matched:
                matches[index] = matched
        return matches


def shared_ngrams(
    corpus: Sequence[str], items_with: dict[NGram, set[int]], n: int, tokenizer: Tokenizer
) -> dict[int, set[NGram]]:
    """For each document, the benchmark n-grams it contains."""
    shared: dict[int, set[NGram]] = {}
    for index, document in enumerate(corpus):
        if document.strip():
            found = {gram for gram in ngrams(tokenizer(document), n) if gram in items_with}
            if found:
                shared[index] = found
    return shared


@dataclass(frozen=True)
class RareSharedNgram(MatchingRule):
    """Flag an item on a shared n-gram only if few training documents contain it.

    The false positives of n-gram filters come from conventional phrasing, such
    as "let m and n be relatively prime positive integers", which appears in
    many documents. The spans specific to one problem appear in few. Ignoring
    n-grams found in more than `max_documents` documents drops the convention
    and keeps the problem. Document frequency is counted on the corpus being
    filtered, so the rule needs no list or labels.
    """

    n: int
    tokenizer: Tokenizer
    max_documents: int

    def match(self, corpus, benchmark):
        items_with: dict[NGram, set[int]] = {}
        for item, text in enumerate(benchmark):
            for gram in ngrams(self.tokenizer(text), self.n):
                items_with.setdefault(gram, set()).add(item)

        shared = shared_ngrams(corpus, items_with, self.n, self.tokenizer)
        frequency = Counter(gram for grams in shared.values() for gram in grams)
        matches: Matches = {}
        for index, grams in shared.items():
            matched = set().union(
                *(items_with[gram] for gram in grams if frequency[gram] <= self.max_documents)
            )
            if matched:
                matches[index] = matched
        return matches


@dataclass(frozen=True)
class AnyOf(MatchingRule):
    """Flag an item if any of the component rules does."""

    rules: tuple[MatchingRule, ...]

    def match(self, corpus, benchmark):
        combined: Matches = {}
        for rule in self.rules:
            for document, items in rule.match(corpus, benchmark).items():
                combined.setdefault(document, set()).update(items)
        return combined

    def __or__(self, other: MatchingRule) -> AnyOf:
        return AnyOf((*self.rules, other))
