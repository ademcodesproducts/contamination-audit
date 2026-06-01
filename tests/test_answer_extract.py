"""Answer extraction + equivalence.

The DiD accuracy numbers in Table 5 depend on extracting and equivalence-checking
``\\boxed{}`` answers. A regression here silently rewrites the paper's
contaminated-vs-clean accuracy gap, so the most common patterns are pinned.
"""

from contamination_audit.answer_extract import (
    equivalent,
    extract_boxed,
    extract_from_trace,
    normalize,
)


def test_extract_boxed_returns_last():
    trace = r"First try: \boxed{5}. Wait, recomputing... \boxed{8}."
    assert extract_boxed(trace) == "8"


def test_extract_boxed_returns_none_when_absent():
    """Null returns drive the null-rate metric in Table 6 / Figure 1."""
    assert extract_boxed("no boxed answer here") is None


def test_extract_from_trace_falls_back_to_natural_language():
    trace = "After much work, the answer is 42."
    assert extract_from_trace(trace) == "42"


def test_equivalent_fraction_and_decimal():
    """1/2 must equal 0.5 — this is one of the example flips in Appendix E."""
    assert equivalent("1/2", "0.5")
    assert equivalent(r"\frac{1}{2}", "0.5")


def test_equivalent_degree_marker():
    assert equivalent("30", "30°")
    assert equivalent("30", r"30^\circ")


def test_equivalent_handles_text_wrapper():
    """math500_0301 (Appendix E) was scored wrong because the gt was wrapped in \\text{}."""
    assert equivalent(r"\text{5.4 cents}", "5.4cents")


def test_equivalent_returns_false_for_none():
    assert not equivalent(None, "5")
    assert not equivalent("5", None)


def test_normalize_strips_whitespace():
    assert normalize("  42 ") == "42"
