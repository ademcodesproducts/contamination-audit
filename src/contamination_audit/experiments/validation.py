"""Testing the assumption the control set rests on.

The method assumes nothing published after the corpus was assembled can be in
it. Dolci remixes separately curated datasets, so in principle one component
could postdate the release of the whole. A control item that was genuinely
present would share long spans with some training document; a coincidental
match shares only stock phrasing.

So for every benchmark item this finds the longest word n-gram it shares with
any document, and compares the two groups. The longest control matches are
listed for inspection.
"""

from __future__ import annotations

from collections import Counter

from ..data import DOLCI, BenchmarkItem
from ..filters import S1_WORDS, ngrams
from .base import Experiment

SPAN_LENGTHS = (8, 10, 13, 15, 20, 25, 30, 40)
LONGEST_SHOWN = 5


def longest_shared_spans(
    items: tuple[BenchmarkItem, ...], corpus_tokens: list[list[str]]
) -> tuple[dict[int, int], dict[int, str]]:
    """For each item that shares any span, its longest span length and that span.

    An item usually has several spans tied at its longest length. The
    alphabetically first is kept, so the reported example is the same in every
    run instead of following hash order.
    """
    items_with = {n: {} for n in SPAN_LENGTHS}
    for item, benchmark_item in enumerate(items):
        tokens = S1_WORDS(benchmark_item.problem)
        for n in SPAN_LENGTHS:
            for gram in ngrams(tokens, n):
                items_with[n].setdefault(gram, set()).add(item)

    best_length: dict[int, int] = {}
    best_span: dict[int, str] = {}
    for tokens in corpus_tokens:
        for n in SPAN_LENGTHS:
            for gram in ngrams(tokens, n):
                for item in items_with[n].get(gram, ()):
                    span = " ".join(gram)
                    previous = best_length.get(item)
                    if (
                        previous is None
                        or n > previous
                        or (n == previous and span < best_span[item])
                    ):
                        best_length[item] = n
                        best_span[item] = span
    return best_length, best_span


class ControlValidation(Experiment):
    name = "validation"
    title = "Longest word n-gram each benchmark item shares with the corpus"

    def run(self):
        corpus_tokens = [S1_WORDS(text) for text in self.study.questions(DOLCI) if text.strip()]
        split = self.study.split(DOLCI)

        distribution, longest = {}, {}
        for label, items in (("measurement", split.measurement), ("control", split.control)):
            best_length, best_span = longest_shared_spans(items, corpus_tokens)
            counts = Counter(best_length.get(item, 0) for item in range(len(items)))
            distribution[label] = {str(n): count for n, count in sorted(counts.items())}

            top = sorted(best_length, key=lambda item: (-best_length[item], item))
            longest[label] = [
                {
                    "item": items[item].id,
                    "benchmark": items[item].benchmark,
                    "n": best_length[item],
                    "span": best_span[item],
                }
                for item in top[:LONGEST_SHOWN]
            ]
        return {"distribution": distribution, "longest": longest}

    def report(self, result):
        lines = []
        for label, counts in result["distribution"].items():
            summary = ", ".join(
                f"{'none' if n == '0' else '>=' + n}: {count}" for n, count in counts.items()
            )
            lines.append(f"{label}: {summary}")
        lines.append("longest control spans, which should be stock phrasing:")
        for row in result["longest"]["control"]:
            lines.append(f'  n={row["n"]:2d}  {row["benchmark"]:18s} "{row["span"][:70]}"')
        return lines
