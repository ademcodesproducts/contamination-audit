"""An embedding filter, measured the same way as the string filters.

The retrieval stage of the LLM Decontaminator (filters/embedding.py), with a
threshold on cosine similarity in place of its LLM verdict. Every threshold
gets the same treatment as the string filters: false positives on the
controls, flag rates by kind of question, the matched comparison of each
competition with its 2026 edition, the training documents it would remove,
and recall on the planted copies, paraphrases included.
"""

from __future__ import annotations

from ..data import DOLCI
from ..filters.embedding import MODEL, MODEL_REVISION, Encoder, nearest
from .base import Experiment
from .matched import SERIES
from .planted import planted_documents
from .rates import bootstrap_rate, matched_comparison
from .strata import kind_of

THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


class EmbeddingFilter(Experiment):
    name = "embedding"
    title = "An embedding filter on all of Dolci: retrieval as in the LLM Decontaminator"

    def run(self):
        encoder = Encoder()
        work = self.study.paths.work / "embeddings"
        split = self.study.split(DOLCI)
        items = list(self.study.scan_items)
        position = {item.id: i for i, item in enumerate(items)}
        measure = [position[i.id] for i in split.measurement]
        control = [position[i.id] for i in split.control]

        corpus = encoder.encode_cached(self.study.full_questions(DOLCI), work, "dolci")
        vectors = encoder.encode([item.problem for item in items])
        found = nearest(corpus, vectors, {"measure": measure, "control": control})

        by_edition: dict[str, list[int]] = {}
        for item in items:
            by_edition.setdefault(item.benchmark, []).append(position[item.id])
        later_sizes = [len(by_edition[later]) for _, later in SERIES.values()]
        weights = [size / sum(later_sizes) for size in later_sizes]

        planted_control, planted, forms = planted_documents(self.study)
        planted_vectors = encoder.encode([planted_control[i].problem for i in planted])
        # Recall counts an item when one of the planted documents retrieves it,
        # as with the string filters; only planted items count.
        planted_similarity = {
            form: (encoder.encode(documents) @ planted_vectors.T).max(axis=0)
            for form, documents in forms.items()
        }

        sweep = {}
        for threshold in THRESHOLDS:
            hit = found.item_similarity >= threshold

            def flagged(members: list[int], hit=hit) -> set[int]:
                return {k for k, i in enumerate(members) if hit[i]}

            def indicator(edition: str, hit=hit) -> list[int]:
                return [int(hit[i]) for i in by_edition[edition]]

            kinds: dict[str, dict] = {}
            for group, members in (("measure", measure), ("control", control)):
                for kind in sorted({kind_of(items[i]) for i in members}):
                    subset = [i for i in members if kind_of(items[i]) == kind]
                    kinds[f"{group} {kind}"] = bootstrap_rate(flagged(subset), len(subset))

            removed_measure = found.document_similarity["measure"] >= threshold
            removed_control = found.document_similarity["control"] >= threshold
            sweep[f"{threshold:.2f}"] = {
                "measure": bootstrap_rate(flagged(measure), len(measure)),
                "control": bootstrap_rate(flagged(control), len(control)),
                "by_kind": kinds,
                "matched": matched_comparison(
                    [[f for e in earlier for f in indicator(e)] for earlier, _ in SERIES.values()],
                    [indicator(later) for _, later in SERIES.values()],
                    weights,
                ),
                "documents": {
                    "measure": int(removed_measure.sum()),
                    "control": int(removed_control.sum()),
                    "control_only": int((removed_control & ~removed_measure).sum()),
                },
                "planted_recall": {
                    form: bootstrap_rate(
                        {k for k, s in enumerate(similarity) if s >= threshold}, len(planted)
                    )
                    for form, similarity in planted_similarity.items()
                },
            }

        return {
            "model": MODEL,
            "revision": MODEL_REVISION,
            "corpus_n": len(corpus),
            "thresholds": sweep,
            "nearest": {
                item.id: [round(float(found.item_similarity[i]), 4), int(found.item_document[i])]
                for i, item in enumerate(items)
            },
        }

    def report(self, result):
        lines = [
            f"{'cosine':>6s} {'measure':>8s} {'control':>8s} {'excess (pp)':>12s}  "
            "planted recall (verbatim / retypeset / half / prompt / paraphrase)"
        ]
        for threshold, row in result["thresholds"].items():
            recall = row["planted_recall"]
            forms = ("verbatim", "retypeset", "stem", "in_conversation", "paraphrase")
            cells = " / ".join(f"{recall[f]['pct']:.0f}" for f in forms if f in recall)
            lines.append(
                f"{threshold:>6s} {row['measure']['pct']:7.1f}% {row['control']['pct']:7.1f}% "
                f"{row['matched']['excess_pp']['value']:+11.1f}  {cells}"
            )
        return lines
