"""Flag rates and bootstrap intervals."""

from contamination_audit.experiments.rates import (
    Rate,
    bootstrap_rate,
    central_interval,
    false_discovery_share,
    matched_comparison,
)


def test_rate_as_fraction_percent_and_text():
    rate = Rate(36, 280)
    assert rate.fraction == 36 / 280
    assert rate.percent == 100 * 36 / 280
    assert str(rate) == "36 (12.9%)"


def test_false_discovery_share_is_the_ratio_of_rates():
    assert false_discovery_share(Rate(36, 280), Rate(127, 502)) == (36 / 280) / (127 / 502)


def test_false_discovery_share_is_undefined_when_nothing_is_flagged():
    assert false_discovery_share(Rate(3, 280), Rate(0, 502)) is None


def test_central_interval_drops_the_outer_two_and_a_half_percent():
    assert central_interval(range(1000)) == (25, 975)


def test_bootstrap_is_seeded_and_brackets_the_rate():
    flagged = set(range(0, 502, 4))
    first, second = bootstrap_rate(flagged, 502), bootstrap_rate(flagged, 502)
    assert first == second
    low, high = first["ci"]
    assert low < first["pct"] < high


def test_bootstrap_of_nothing_flagged_is_a_point():
    assert bootstrap_rate(set(), 280) == {"pct": 0.0, "ci": [0.0, 0.0]}


def test_matched_comparison_of_identical_groups_shows_no_excess():
    cells = [[1, 0, 0, 1], [0, 1]]
    result = matched_comparison(cells, cells, [0.5, 0.5])
    assert result["excess_pp"]["value"] == 0
    assert result["share"]["value"] == 1


def test_matched_comparison_weights_series_by_the_later_editions():
    earlier, later = [[1, 1], [0, 0]], [[1, 0], [0, 0]]
    result = matched_comparison(earlier, later, [0.25, 0.75])
    assert result["earlier_pct"] == 25.0
    assert result["later_pct"] == 12.5
