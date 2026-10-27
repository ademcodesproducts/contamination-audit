"""The matching rules behave as their source projects' code does.

These pin the behaviour that distinguishes the filters from one another, since
that distinction is what the paper measures. They are deliberately small: the
point is fidelity to each project's code, not detecting contamination well.
"""

import pytest

from contamination_audit.filters import (
    S1_WORDS,
    AnyOf,
    FuzzyRatio,
    PhraseAgreement,
    SharedNgram,
    TokenCoverage,
    ngrams,
)

FOX = "the quick brown fox jumps over the lazy dog"


def words(n: int, start: int = 0) -> str:
    """'w000 w001 ...': distinct tokens, so every n-gram is unique."""
    return " ".join(f"w{i:03d}" for i in range(start, start + n))


# ------------------------------------------------------------------ tokenizer


def test_s1_words_lowercase_but_keep_punctuation():
    # LaTeX stays attached to its word; stripping it would change what an 8-gram is.
    assert S1_WORDS("Let $f(x)$  BE") == ["let", "$f(x)$", "be"]


@pytest.mark.parametrize("text", ["  Mixed\tCase\nText  ", "", "a  b", "Ünïcode TEXT"])
def test_s1_words_match_s1s_normalize_then_split(text):
    # s1's code collapses whitespace with a join before splitting.
    assert S1_WORDS(text) == " ".join(text.lower().strip().split()).split()


# --------------------------------------------------------------------- ngrams


def test_ngrams_of_a_short_sequence_are_empty():
    assert ngrams(["a", "b"], 3) == set()


def test_ngrams_are_distinct_windows():
    assert ngrams(list("abcab"), 2) == {("a", "b"), ("b", "c"), ("c", "a")}


# ---------------------------------------------------------------- shared ngram


def test_shared_ngram_flags_on_one_shared_window():
    assert SharedNgram(8, S1_WORDS).flag(["nothing here", FOX], [FOX]) == {0}


def test_shared_ngram_ignores_shorter_overlap():
    assert SharedNgram(8, S1_WORDS).flag(["the quick brown"], [FOX]) == set()


def test_blank_documents_are_skipped():
    assert SharedNgram(1, S1_WORDS).flag(["   ", ""], ["anything"]) == set()


# ---------------------------------------------------------------------- fuzzy


def test_fuzzy_ratio_respects_its_threshold():
    item = ["how many integers satisfy this particular inequality"]
    assert FuzzyRatio(95).flag(item, item) == {0}
    assert FuzzyRatio(95).flag(["something entirely unrelated to that"], item) == set()


def test_fuzzy_ratio_takes_the_best_document_for_each_item():
    benchmark = [
        "how many integers satisfy this particular inequality",
        "find the sum of all possible values of the expression",
    ]
    corpus = ["filler", "find the sum of all possible values of the expression", "more"]
    assert FuzzyRatio(95).flag(corpus, benchmark) == {1}


# ------------------------------------------------------------------- coverage


def test_coverage_flags_an_item_covered_by_one_document():
    item = words(10)
    assert TokenCoverage(4, 0.5, S1_WORDS).flag([item], [item]) == {0}
    assert TokenCoverage(4, 0.5, S1_WORDS).flag([words(3)], [item]) == set()


def test_coverage_threshold_is_strict():
    # Tulu 3 says "greater than 50%": one shared 5-gram of a 10-token item
    # covers exactly half, so it must not flag.
    half = [words(5)]
    assert TokenCoverage(5, 0.5, S1_WORDS).flag(half, [words(10)]) == set()
    assert TokenCoverage(5, 0.49, S1_WORDS).flag(half, [words(10)]) == {0}


def test_coverage_is_scored_per_document_not_pooled():
    # Three fragments jointly cover all 20 tokens, but none covers more than 8.
    # open-instruct scores each document separately and reads the best one.
    fragments = [words(8, 0), words(8, 6), words(8, 12)]
    assert TokenCoverage(8, 0.5, S1_WORDS).flag(fragments, [words(20)]) == set()
    assert TokenCoverage(8, 0.5, S1_WORDS).flag([words(20)], [words(20)]) == {0}


# ----------------------------------------------------------- phrase agreement


def test_phrase_agreement_rejects_a_lone_stock_phrase():
    item = FOX + " and then runs far away"
    assert PhraseAgreement().flag([item], [item]) == {0}
    assert PhraseAgreement().flag(["the quick brown fox jumps over the lazy"], [item]) == set()


def test_sampled_phrases_are_spread_through_the_item():
    item = words(60)
    assert PhraseAgreement(0.5).flag([words(20)], [item]) == set()
    assert PhraseAgreement(0.5).flag([item], [item]) == {0}


def test_sampled_phrases_follow_position_not_hash_or_alphabet():
    # Tokens descend alphabetically as they ascend in position, so positional
    # and alphabetical sampling would pick different windows.
    tokens = [f"w{40 - i:03d}" for i in range(40)]
    rule = PhraseAgreement(required_fraction=1.0, phrases=2)
    middle = (len(tokens) - 8 + 1) // 2
    assert rule.sample_phrases(" ".join(tokens)) == [
        tuple(tokens[0:8]),
        tuple(tokens[middle : middle + 8]),
    ]
    document = " ".join(tokens[0:8] + tokens[middle : middle + 8])
    assert rule.flag([document], [" ".join(tokens)]) == {0}


def test_phrase_agreement_is_counted_within_one_document():
    item = words(60)
    halves = [words(30, 0), words(30, 30)]
    assert PhraseAgreement(0.9).flag(halves, [item]) == set()


# ---------------------------------------------------------------- composition


def test_rules_combine_with_or():
    combined = SharedNgram(8, S1_WORDS) | FuzzyRatio(95)
    assert isinstance(combined, AnyOf)
    corpus = [FOX, "find the sum of all possible values of the expression"]
    benchmark = ["prefix " + FOX, "find the sum of all possible values of the expression"]
    assert combined.flag(corpus, benchmark) == {0, 1}


def test_chained_or_stays_flat():
    a, b, c = SharedNgram(5, S1_WORDS), SharedNgram(8, S1_WORDS), FuzzyRatio(90)
    assert (a | b | c).rules == (a, b, c)


@pytest.mark.parametrize(
    "rule",
    [
        SharedNgram(3, S1_WORDS),
        FuzzyRatio(80),
        TokenCoverage(3, 0.5, S1_WORDS),
        PhraseAgreement(0.5, n=3, phrases=4),
        SharedNgram(4, S1_WORDS) | FuzzyRatio(80),
    ],
)
def test_flagging_groups_together_equals_flagging_them_apart(rule):
    # Experiments flag measurement and control in one pass, which is only valid
    # because every rule decides each benchmark item independently.
    corpus = [words(12, 0), words(6, 30), "a b c d e f", words(9, 50)]
    first = [words(6, 2), words(6, 100), "a b c d e f"]
    second = [words(8, 50), words(4, 31), "unrelated text entirely"]
    assert rule.flag_groups(corpus, first, second) == [
        rule.flag(corpus, first),
        rule.flag(corpus, second),
    ]
