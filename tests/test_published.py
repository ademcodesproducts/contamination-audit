"""The registry of published filters."""

from contamination_audit.filters import Provenance, published_filters


def test_order_matches_the_papers_tables():
    # Results are reported in registry order, so the order is part of what the
    # tables mean.
    assert [f.name for f in published_filters()] == [
        "s1-8gram-word",
        "ot3-13gram-qwen-or-fuzz75",
        "ot114k-fuzz95",
        "tulu3-8gram-50pct",
    ]


def test_partial_reimplementations_are_excluded_by_default():
    # decon's candidate stage alone flags 199 of 200 items where decon flags 0.
    assert all(f.faithful for f in published_filters())
    assert any(not f.faithful for f in published_filters(include_partial=True))


def test_every_filter_records_where_its_parameters_came_from():
    assert {f.provenance for f in published_filters()} <= set(Provenance)
    assert Provenance("prose") in {f.provenance for f in published_filters()}
