"""The decontamination filters released with open post-training datasets.

Each filter records where its parameters come from, because that provenance is
part of the result: some are fixed by their project's released code, one only
by its paper's prose, and the filters do not agree.

Ai2's decon is the fifth filter in the paper. It is a Rust binary rather than a
rule reimplemented here; see `contamination_audit.decon`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from .rules import FuzzyRatio, MatchingRule, SharedNgram, TokenCoverage
from .tokenizers import CL100K, QWEN2, S1_WORDS, SPACY_LG


class Provenance(str, Enum):
    CODE = "code"  # parameters are defaults in the project's released source
    PROSE = "prose"  # parameters are stated only in the project's paper


@dataclass(frozen=True)
class PublishedFilter:
    name: str
    project: str
    provenance: Provenance
    rule: MatchingRule
    description: str
    # A partial reimplementation must never sit in a table beside complete
    # ones, where its number would be read as that project's behaviour.
    faithful: bool = True

    def flag(self, corpus: Sequence[str], benchmark: Sequence[str]) -> set[int]:
        return self.rule.flag(corpus, benchmark)

    def flag_groups(self, corpus: Sequence[str], *groups: Sequence[str]) -> list[set[int]]:
        return self.rule.flag_groups(corpus, *groups)


# OpenThoughts3's n-gram branch on its own. Several experiments isolate it,
# since the fuzzy branch adds only three items on the main corpus.
OPENTHOUGHTS3_NGRAM = SharedNgram(13, QWEN2)

S1 = PublishedFilter(
    name="s1-8gram-word",
    project="s1",
    provenance=Provenance.CODE,
    rule=SharedNgram(8, S1_WORDS),
    description="Any shared 8-gram of lowercased whitespace tokens.",
)

OPENTHOUGHTS3 = PublishedFilter(
    name="ot3-13gram-qwen-or-fuzz75",
    project="OpenThoughts3",
    provenance=Provenance.CODE,
    rule=OPENTHOUGHTS3_NGRAM | FuzzyRatio(75.0),
    description=(
        "Any shared 13-gram of Qwen2 tokens, or a fuzz.ratio of at least 75. Both "
        "branches, as the paper states ('marked as contaminated by either of our "
        "methods'). The released math config sets the fuzzy cutoff to 80, which "
        "adds no items on this corpus."
    ),
)

OPENTHOUGHTS_114K = PublishedFilter(
    name="ot114k-fuzz95",
    project="OpenThoughts-114K",
    provenance=Provenance.CODE,
    rule=FuzzyRatio(95.0),
    description="A fuzz.ratio of at least 95 on raw strings.",
)

TULU3 = PublishedFilter(
    name="tulu3-8gram-50pct",
    project="Tulu 3",
    provenance=Provenance.PROSE,
    rule=TokenCoverage(n=8, threshold=0.5, tokenizer=SPACY_LG),
    description=(
        "8-grams of spaCy tokens covering more than half of an item within one "
        "training document. The tokenizer and per-document scoring follow "
        "open-instruct; n and the threshold come from the paper, since the released "
        "code declares no defaults for either. open-instruct finds candidate "
        "documents through Elasticsearch, capped at 100 hits per n-gram, where this "
        "scans the corpus exhaustively. Seeing every matching document can only "
        "raise an item's best score, so this is an upper bound on the original."
    ),
)

DECON_CANDIDATES_ONLY = PublishedFilter(
    name="PARTIAL-cl100k-5gram-candidates-only",
    project="NOT decon",
    provenance=Provenance.CODE,
    rule=SharedNgram(5, CL100K),
    faithful=False,
    description=(
        "decon's candidate-generation stage alone: any shared 5-gram of cl100k "
        "tokens. decon then expands clusters and keeps candidates whose weighted "
        "score clears 0.8, which is not implemented here. On 100,000 Dolci "
        "questions this flags 199 of the 200 OlymMATH items, where the real decon "
        "binary flags none. Kept only to show that a two-stage filter cannot be "
        "approximated by its first stage."
    ),
)

_ALL = (S1, OPENTHOUGHTS3, OPENTHOUGHTS_114K, TULU3, DECON_CANDIDATES_ONLY)


def published_filters(include_partial: bool = False) -> list[PublishedFilter]:
    """The filters in the order the paper's tables list them.

    Partial reimplementations are left out unless asked for, and their output
    is for diagnosis only.
    """
    return [f for f in _ALL if f.faithful or include_partial]
