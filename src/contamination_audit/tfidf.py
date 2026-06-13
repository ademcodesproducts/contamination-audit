"""TF-IDF semantic retrieval (Stage 2 alternative).

Paper §5.1 specifies TF-IDF cosine similarity for the **Tülu 3** detection
pipeline (dense embeddings for OpenThoughts). The teammate implementation in
``downloads/shortcuts-not-recall-main/src/02_detect_contamination.py`` uses:

- character n-grams (3–5) — handles LaTeX notation precisely
- ``sublinear_tf=True`` and ``max_features=100_000``
- a density-aware top-K retrieval that distinguishes "spike" (instance
  contamination — one close match) from "plateau" (template contamination —
  many near matches) patterns

This module ports that retrieval logic into the library so it can be selected
per project via ``configs/thresholds.yaml``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TfidfConfig:
    analyzer: str = "char_wb"
    ngram_range: tuple[int, int] = (3, 5)
    max_features: int = 100_000
    sublinear_tf: bool = True
    density_threshold: float = 0.5
    top_k: int = 5


def build_vectorizer(config: TfidfConfig):
    """Fit a TF-IDF vectorizer; lazy-imports sklearn so the library stays slim."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(
        analyzer=config.analyzer,
        ngram_range=config.ngram_range,
        max_features=config.max_features,
        sublinear_tf=config.sublinear_tf,
    )


def retrieve(
    *,
    project: str,
    train_items: list[dict],
    math500_items: list[dict],
    config: TfidfConfig,
) -> list[dict]:
    """Run TF-IDF retrieval, returning one candidate record per (math500, top-train) pair.

    The record schema mirrors ``embedding.search_candidates`` so that downstream
    stages (LLM judge, validation) can consume either retrieval method.
    """
    train_texts = [item.get("problem", "") for item in train_items]
    math_texts = [item.get("problem", "") for item in math500_items]

    _log.info(
        "tfidf retrieval: project=%s train=%d math500=%d",
        project, len(train_texts), len(math_texts),
    )

    vectorizer = build_vectorizer(config)
    vectorizer.fit(train_texts + math_texts)
    train_vecs = vectorizer.transform(train_texts)
    math_vecs = vectorizer.transform(math_texts)

    # Sparse @ sparse.T → dense (500 × N is manageable for MATH-500 scale).
    sim_matrix = (math_vecs @ train_vecs.T).toarray()

    candidates: list[dict] = []
    for m_idx, math_item in enumerate(math500_items):
        sims = sim_matrix[m_idx]
        # Top-K neighbors by similarity (descending)
        top_idx = np.argsort(sims)[-config.top_k :][::-1]
        n_above_thresh = int((sims > config.density_threshold).sum())
        top_sims = [round(float(sims[i]), 4) for i in top_idx]
        for rank, nbr in enumerate(top_idx):
            train_item = train_items[int(nbr)]
            candidates.append({
                "project": project,
                "contamination_type": "c_sem_candidate",
                "math500_id": math_item["math500_id"],
                "math500_problem": math_item.get("problem", ""),
                "math500_answer": math_item.get("answer", ""),
                "math500_subject": math_item.get("subject", ""),
                "math500_level": math_item.get("level", -1),
                "train_id": train_item["train_id"],
                "train_problem": train_item.get("problem", ""),
                "train_solution": (train_item.get("solution") or "")[:500],
                "similarity_score": float(sims[int(nbr)]),
                "tfidf_rank": rank,
                "tfidf_n_neighbors_above_threshold": n_above_thresh,
                "tfidf_top_k_sims": top_sims,
                "retrieval_method": "tfidf",
            })

    candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
    unique_math = len({c["math500_id"] for c in candidates})
    _log.info(
        "  %d candidates (%d unique MATH-500, %d with density>=%.2f)",
        len(candidates), unique_math,
        sum(1 for c in candidates if c["tfidf_n_neighbors_above_threshold"] >= 1),
        config.density_threshold,
    )
    return candidates
