"""Which parts of Dolci the OpenThoughts3 n-gram flags come from.

Dolci is a remix of separately curated datasets. Each flag is attributed to the
source of the document that triggered it, and compared with that source's
share of the corpus: an enrichment above 1 means a source produces more flags
than its size explains.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ..data import DOLCI
from ..filters import OPENTHOUGHTS3_NGRAM, ngrams
from .base import Experiment


class Composition(Experiment):
    name = "composition"
    title = "Dolci sources of the OpenThoughts3 n-gram's measurement flags"

    def run(self):
        documents = self.study.documents(DOLCI)
        measurement = self.study.split(DOLCI).measurement_texts
        n, tokenize = OPENTHOUGHTS3_NGRAM.n, OPENTHOUGHTS3_NGRAM.tokenizer

        items_with: dict[tuple, set[int]] = {}
        for item, text in enumerate(measurement):
            for gram in ngrams(tokenize(text), n):
                items_with.setdefault(gram, set()).add(item)

        source_sizes = Counter(document.source for document in documents)
        flagging_documents: Counter[str] = Counter()
        items_flagged: defaultdict[str, set[int]] = defaultdict(set)
        for document in documents:
            if not document.problem.strip():
                continue
            hit: set[int] = set()
            for gram in ngrams(tokenize(document.problem), n):
                matched = items_with.get(gram)
                if matched:
                    hit |= matched
            if hit:
                flagging_documents[document.source] += 1
                items_flagged[document.source] |= hit

        n_flagged = len(set().union(*items_flagged.values()))
        sources = []
        for source, n_documents in flagging_documents.most_common():
            corpus_share = source_sizes[source] / len(documents)
            flag_share = len(items_flagged[source]) / n_flagged if n_flagged else 0
            sources.append(
                {
                    "source": source,
                    "corpus_share": 100 * corpus_share,
                    "docs": n_documents,
                    "items": len(items_flagged[source]),
                    "enrichment": flag_share / corpus_share if corpus_share else None,
                }
            )
        return {"flagged": n_flagged, "slices": sources}

    def report(self, result):
        lines = [f"{'source':44s} {'corpus':>7s} {'docs':>6s} {'items':>6s} {'enrichment':>10s}"]
        for row in result["slices"]:
            lines.append(
                f"{row['source'][:44]:44s} {row['corpus_share']:6.1f}% {row['docs']:6d} "
                f"{row['items']:6d} {row['enrichment']:9.1f}x"
            )
        return lines
