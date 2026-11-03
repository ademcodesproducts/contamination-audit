"""Each competition series against its own edition that postdates the corpus.

Control and measurement items should attract spurious matches equally often,
or the control rate says little about the measurement set. The cleanest way to
get that is to hold the competition fixed: AIME 2023 to 2025 and AIME 2026 are
written to the same conventions, and only the earlier editions can be in the
corpus. Whatever a filter flags on AIME 2026 is noise, so its excess on the
earlier editions is the most it can be detecting.
"""

from __future__ import annotations

from ..data import BenchmarkItem
from ..errors import MissingDataError
from .base import Experiment
from .rates import Rate, matched_comparison

# Each series: its editions that predate Dolci, and its edition that postdates it.
SERIES = {
    "aime": (
        ("aime_2023_I", "aime_2023_II", "aime_2024_I", "aime_2024_II", "aime_2025"),
        "aime_2026",
    ),
    "hmmt_feb": (("hmmt_feb_2023", "hmmt_feb_2024", "hmmt_feb_2025"), "hmmt_feb_2026"),
    "usamo": (("usamo_2024", "usamo_2025"), "usamo_2026"),
}


def pooled(indicator: dict[str, list[int]], editions: tuple[str, ...]) -> list[int]:
    """The flag indicators of several editions, as one group."""
    return [flag for edition in editions for flag in indicator[edition]]


class MatchedCompetitions(Experiment):
    name = "matched"
    title = "Each competition series against its 2026 edition"

    def run(self):
        library = self.study.benchmarks
        by_edition: dict[str, list[BenchmarkItem]] = {}
        for item in library.measurement + library.control + library.earlier_editions:
            by_edition.setdefault(item.benchmark, []).append(item)

        editions = [e for earlier, later in SERIES.values() for e in (*earlier, later)]
        missing = [e for e in editions if e not in by_edition]
        if missing:
            raise MissingDataError(f"no benchmark items for {', '.join(missing)}")

        later_sizes = {series: len(by_edition[later]) for series, (_, later) in SERIES.items()}
        weights = [size / sum(later_sizes.values()) for size in later_sizes.values()]

        filters = {}
        for published in self.study.filters:
            scan = self.study.scan(published)
            flags = {e: scan.flagged(by_edition[e]) for e in editions}
            indicator = {
                e: [1 if i in flags[e] else 0 for i in range(len(by_edition[e]))] for e in editions
            }

            filters[published.name] = {
                "editions": {
                    e: {
                        "n": len(indicator[e]),
                        "pct": Rate(sum(indicator[e]), len(indicator[e])).percent,
                    }
                    for e in editions
                },
                "series": {
                    series: matched_comparison(
                        [pooled(indicator, earlier)], [indicator[later]], [1.0]
                    )
                    for series, (earlier, later) in SERIES.items()
                },
                "pooled": matched_comparison(
                    [pooled(indicator, earlier) for earlier, _ in SERIES.values()],
                    [indicator[later] for _, later in SERIES.values()],
                    weights,
                ),
            }

        return {
            "series": {
                series: {"earlier": list(earlier), "later": later}
                for series, (earlier, later) in SERIES.items()
            },
            "weights": dict(zip(SERIES, weights, strict=True)),
            "filters": filters,
        }

    def report(self, result):
        header = f"{'filter':28s} {'series':9s} {'earlier':>8s} {'2026':>7s}"
        lines = [f"{header} {'excess (pp), 95% CI':>24s}"]
        for name, row in result["filters"].items():
            comparisons = {**row["series"], "pooled": row["pooled"]}
            for series, c in comparisons.items():
                low, high = c["excess_pp"]["ci"]
                lines.append(
                    f"{name:28s} {series:9s} {c['earlier_pct']:7.1f}% {c['later_pct']:6.1f}% "
                    f"{c['excess_pp']['value']:+8.1f} [{low:+.1f}, {high:+.1f}]"
                )
        return lines
