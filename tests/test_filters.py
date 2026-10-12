"""Tests for the filter implementations the paper's results depend on.

These pin the behaviour that distinguishes the filters from one another, since
that distinction is the paper's subject. They are deliberately small: the point
is that each filter does what its source project's code does, not that it
detects contamination well.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from contamination_audit.filters import (
    flag_by_coverage,
    flag_by_fuzz,
    flag_by_ngram,
    flag_by_sampled_phrases,
    ngrams,
    registry,
    tok_alnum,
    tok_s1,
    tok_whitespace,
)

# ----------------------------------------------------------------- tokenizers


def test_s1_tokenizer_lowercases_but_keeps_punctuation():
    # s1 normalizes with lower().strip() and whitespace collapse only, so LaTeX
    # survives as part of the token. Stripping it would change what an 8-gram is.
    assert tok_s1("Let $f(x)$  BE") == ["let", "$f(x)$", "be"]


def test_whitespace_tokenizer_preserves_case():
    assert tok_whitespace("Let $f(x)$ be") == ["Let", "$f(x)$", "be"]


def test_tokenizers_differ_on_punctuation():
    # The choice no published spec states, and the one that moves the result most.
    assert tok_alnum("a+b") == ["a", "b"]
    assert tok_whitespace("a+b") == ["a+b"]


# ----------------------------------------------------------------- primitives


def test_ngrams_shorter_than_n_yields_nothing():
    assert ngrams(["a", "b"], 3) == set()


def test_ngrams_counts_windows():
    assert len(ngrams(list("abcde"), 3)) == 3


# -------------------------------------------------------------------- filters


def test_ngram_filter_flags_on_a_single_shared_window():
    bench = ["the quick brown fox jumps over the lazy dog"]
    corpus = ["nothing here", "the quick brown fox jumps over the lazy dog"]
    assert flag_by_ngram(corpus, bench, 8, tok_s1) == {0}


def test_ngram_filter_ignores_short_overlap():
    bench = ["the quick brown fox jumps over the lazy dog"]
    assert flag_by_ngram(["the quick brown"], bench, 8, tok_s1) == set()


def test_fuzz_filter_respects_its_cutoff():
    bench = ["how many integers satisfy this particular inequality"]
    assert flag_by_fuzz(bench, bench, 95.0) == {0}
    assert flag_by_fuzz(["something entirely unrelated to that"], bench, 95.0) == set()


def test_fuzz_filter_takes_the_best_match_over_the_whole_corpus():
    # It reduces a corpus-by-benchmark score matrix along the corpus axis; a
    # single match anywhere must flag, and must flag the right item.
    bench = [
        "how many integers satisfy this particular inequality",
        "find the sum of all possible values of the expression",
    ]
    corpus = [
        "unrelated filler",
        "find the sum of all possible values of the expression",
        "more filler",
    ]
    assert flag_by_fuzz(corpus, bench, 95.0) == {1}


def test_coverage_filter_needs_more_than_the_fraction():
    bench = ["alpha beta gamma delta epsilon zeta eta theta iota kappa"]
    # The benchmark item against itself is fully covered.
    assert flag_by_coverage(bench, bench, 4, 0.5, tok_whitespace) == {0}
    assert (
        flag_by_coverage(["alpha beta gamma"], bench, 4, 0.5, tok_whitespace) == set()
    )


def test_coverage_threshold_is_strict():
    # Tulu 3's paper says "greater than 50%", so exactly half must not flag. Ten
    # tokens, one shared 5-gram, covers five of them: at the boundary and out.
    bench = ["alpha beta gamma delta epsilon zeta eta theta iota kappa"]
    half = ["alpha beta gamma delta epsilon"]
    assert flag_by_coverage(half, bench, 5, 0.5, tok_whitespace) == set()
    assert flag_by_coverage(half, bench, 5, 0.49, tok_whitespace) == {0}


def test_phrase_agreement_rejects_a_lone_stock_phrase():
    # The mechanism the paper identifies: one shared conventional phrase should
    # not be enough once agreement across spans is required.
    bench = ["the quick brown fox jumps over the lazy dog and then runs far away"]
    assert flag_by_sampled_phrases(bench, bench) == {0}
    assert (
        flag_by_sampled_phrases(["the quick brown fox jumps over the lazy"], bench)
        == set()
    )


# -------------------------------------------------------------------- registry


def test_partial_reimplementations_are_excluded_by_default():
    # A partial reimplementation must never appear beside complete ones: decon's
    # candidate stage alone flags 199/200 items where the real binary flags 0.
    assert all(f.faithful for f in registry())
    assert any(not f.faithful for f in registry(include_partial=True))


def test_every_filter_records_where_its_parameters_came_from():
    assert {f.provenance for f in registry()} <= {"code", "prose"}


def test_registry_order_matches_the_paper_tables():
    # scripts/02_results.py reports filters in registry order, so the order is
    # part of what the tables mean. Changing it reorders every table row.
    assert [f.name for f in registry()] == [
        "s1-8gram-word",
        "ot3-13gram-qwen-or-fuzz75",
        "ot114k-fuzz95",
        "tulu3-8gram-50pct",
    ]


def test_coverage_is_scored_per_document_not_pooled():
    # open-instruct scores each training document separately and reads the best
    # one (decontamination/search.py:102,133). Pooling across documents let
    # three unrelated documents jointly "cover" an item none of them contained.
    item = " ".join(f"w{i}" for i in range(20))
    fragments = [
        " ".join(f"w{i}" for i in range(0, 8)),
        " ".join(f"w{i}" for i in range(6, 14)),
        " ".join(f"w{i}" for i in range(12, 20)),
    ]
    # Together the fragments cover all 20 tokens; separately each covers 8.
    assert flag_by_coverage(fragments, [item], 8, 0.5, tok_whitespace) == set()
    assert flag_by_coverage([item], [item], 8, 0.5, tok_whitespace) == {0}


def test_sampled_phrases_are_spread_through_the_item_not_alphabetised():
    # Sorting the n-grams before sampling spaced them through an alphabetised
    # list, so the positional spread the method depends on was never applied.
    tokens = [f"w{i:03d}" for i in range(60)]
    item = " ".join(tokens)
    # A document holding only the item's opening should not satisfy a sample
    # that is genuinely spread across the whole item.
    opening = " ".join(tokens[:20])
    assert flag_by_sampled_phrases([opening], [item], frac=0.5) == set()
    assert flag_by_sampled_phrases([item], [item], frac=0.5) == {0}


def test_sampled_phrases_are_the_positional_ones():
    # `ngrams` returns a set, so sampling from it spaces the phrases through a
    # hash or alphabetical ordering rather than through the problem. The tokens
    # here descend alphabetically as they ascend positionally, so the two
    # orderings select different windows and the distinction is visible.
    tokens = [f"w{40 - i:03d}" for i in range(40)]
    item = " ".join(tokens)

    # With n_phrases=2 the positional sample is the first window and the
    # middle one. A document holding exactly those two satisfies frac=1.0.
    n_windows = len(tokens) - 8 + 1
    positional = [tokens[0:8], tokens[n_windows // 2 : n_windows // 2 + 8]]
    document = " ".join(" ".join(w) for w in positional)

    assert flag_by_sampled_phrases(
        [document], [item], n_phrases=2, frac=1.0
    ) == {0}, "the two sampled phrases should be the first and middle windows"


def test_phrase_agreement_is_counted_within_one_document():
    # Phrases satisfied by several unrelated documents are exactly the
    # stock-phrase match this filter exists to suppress.
    tokens = [f"w{i:03d}" for i in range(60)]
    item = " ".join(tokens)
    halves = [" ".join(tokens[:30]), " ".join(tokens[30:])]
    assert flag_by_sampled_phrases(halves, [item], frac=0.9) == set()
