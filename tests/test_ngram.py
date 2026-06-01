"""N-gram extraction + coverage threshold logic.

Uses synthetic token sequences to keep the test independent of the heavy Qwen2
tokenizer. ``find_hits`` itself is integration-tested via the dedup-by-math500
filter, which exercises the same code path.
"""

from contamination_audit.ngram import dedup_by_math500, extract_ngrams, token_coverage


def test_extract_ngrams_basic():
    tokens = [1, 2, 3, 4, 5]
    assert extract_ngrams(tokens, 3) == {(1, 2, 3), (2, 3, 4), (3, 4, 5)}


def test_extract_ngrams_too_short():
    """Token sequences shorter than n produce no n-grams (matches Qwen behaviour)."""
    assert extract_ngrams([1, 2], 3) == set()
    assert extract_ngrams([], 5) == set()


def test_token_coverage_any_overlap():
    """Tülu 3 metric: covered indices divided by test length, with overlap counted once."""
    test_tokens = [10, 20, 30, 40, 50, 60, 70, 80]
    shared = {(20, 30, 40)}
    coverage = token_coverage(test_tokens, shared, n=3)
    assert coverage == 3 / 8


def test_token_coverage_zero_when_no_match():
    test_tokens = [10, 20, 30]
    coverage = token_coverage(test_tokens, set(), n=3)
    assert coverage == 0.0


def test_dedup_by_math500_keeps_strongest_per_id():
    """Conservative C_lex selection: max shared count per math500_id, threshold applied."""
    hits = [
        {"math500_id": "math500_0001", "n_shared_ngrams": 3},
        {"math500_id": "math500_0001", "n_shared_ngrams": 7},  # winner
        {"math500_id": "math500_0002", "n_shared_ngrams": 5},
        {"math500_id": "math500_0003", "n_shared_ngrams": 1},  # filtered out at min=5
    ]
    unique = dedup_by_math500(hits, min_shared=5)
    assert {item["math500_id"] for item in unique} == {"math500_0001", "math500_0002"}
    winner = next(item for item in unique if item["math500_id"] == "math500_0001")
    assert winner["n_shared_ngrams"] == 7


def test_dedup_by_math500_raw_threshold():
    """min_shared=1 should keep every distinct math500_id at least once."""
    hits = [
        {"math500_id": "math500_0001", "n_shared_ngrams": 1},
        {"math500_id": "math500_0002", "n_shared_ngrams": 1},
    ]
    unique = dedup_by_math500(hits, min_shared=1)
    assert len(unique) == 2
