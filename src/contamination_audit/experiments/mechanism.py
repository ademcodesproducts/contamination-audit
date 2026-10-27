"""The shared phrases behind s1's false positives.

Every control flag is wrong, so the n-grams that trigger them show what the
filter is actually matching on. They turn out to be a few dozen conventional
phrases of competition mathematics, each present in many training documents.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ..data import DOLCI
from ..filters import S1, ngrams
from .base import Experiment

TOP_PHRASES = 15


class Mechanism(Experiment):
    name = "mechanism"
    title = "The 8-grams that cause s1's false positives"

    def run(self):
        corpus = self.study.questions(DOLCI)
        control = self.study.split(DOLCI).control_texts
        n, tokenize = S1.rule.n, S1.rule.tokenizer

        items_with: dict[tuple, set[int]] = {}
        for item, text in enumerate(control):
            for gram in ngrams(tokenize(text), n):
                items_with.setdefault(gram, set()).add(item)

        # For each shared n-gram: how many documents contain it, and which
        # control items it therefore flags.
        documents_with: Counter[tuple] = Counter()
        items_hit: defaultdict[tuple, set[int]] = defaultdict(set)
        for document in corpus:
            if not document.strip():
                continue
            for gram in ngrams(tokenize(document), n):
                hit = items_with.get(gram)
                if hit:
                    documents_with[gram] += 1
                    items_hit[gram] |= hit

        # Ties are broken on the phrase itself, so the ranking is identical
        # between runs rather than following hash order.
        ranked = sorted(documents_with.items(), key=lambda pair: (-pair[1], pair[0]))
        return {
            "distinct_ngrams": len(documents_with),
            "false_positives": len(set().union(*items_hit.values())),
            "n_control": len(control),
            "top": [
                {"ngram": " ".join(gram), "docs": count, "items": len(items_hit[gram])}
                for gram, count in ranked[:TOP_PHRASES]
            ],
        }

    def report(self, result):
        lines = [
            f"{result['false_positives']} of {result['n_control']} controls flagged, "
            f"tracing to {result['distinct_ngrams']} distinct 8-grams"
        ]
        for row in result["top"]:
            lines.append(f'{row["docs"]:5d} documents, {row["items"]:2d} items  "{row["ngram"]}"')
        return lines
