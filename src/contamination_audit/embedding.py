"""Dense semantic retrieval (Stage 2 of the detection pipeline).

For each MATH-500 problem, find the top-K most similar training items using
``all-mpnet-base-v2`` sentence embeddings + FAISS inner-product index, then
filter to similarity ≥ threshold and exclude any pair already flagged by the
n-gram filter (which would be Type 1, not Type 2 contamination).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .config import EmbeddingConfig

_log = logging.getLogger(__name__)


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _empty_gpu_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class Encoder:
    """Lazy wrapper around ``SentenceTransformer`` with chunked + resumable encoding."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = _get_device()
        self._model = None  # loaded lazily

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            _log.info("loading %s on %s", self.config.model, self.device)
            self._model = SentenceTransformer(self.config.model, device=self.device)
        return self._model

    def encode_chunked(self, texts: list[str], cache_path: str | Path) -> np.ndarray:
        """Encode texts with on-disk chunk checkpointing.

        Resumes mid-run from per-chunk ``.npy`` files. After successful completion,
        chunk files are merged into a single ``{cache_path}.npy`` and removed.
        """
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)

        final = cache.with_suffix(".npy")
        if final.exists():
            arr = np.load(final)
            if arr.shape[0] == len(texts):
                _log.info("  cache hit: %s", final)
                return arr

        model = self._ensure_model()
        chunks: list[np.ndarray] = []
        start = 0
        chunk_idx = 0
        n = len(texts)

        while start < n:
            end = min(start + self.config.chunk_size, n)
            chunk_file = cache.parent / f"{cache.stem}_chunk{chunk_idx:04d}.npy"

            if chunk_file.exists():
                arr = np.load(chunk_file)
                if arr.shape[0] == end - start:
                    chunks.append(arr)
                    _log.info("  resume chunk %d (%d/%d)", chunk_idx, end, n)
                    start = end
                    chunk_idx += 1
                    continue

            _log.info("  encoding %d-%d / %d", start, end, n)
            embs = model.encode(
                texts[start:end],
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
            np.save(chunk_file, embs)
            chunks.append(embs)
            _empty_gpu_cache()
            time.sleep(3)  # brief breather between chunks on shared GPUs
            start = end
            chunk_idx += 1

        all_embs = np.concatenate(chunks, axis=0)
        np.save(final, all_embs)
        for f in cache.parent.glob(f"{cache.stem}_chunk*.npy"):
            f.unlink()
        _log.info("  saved %s shape=%s", final, all_embs.shape)
        return all_embs


def build_faiss_index(embeddings: np.ndarray):
    """Inner-product FAISS index. Embeddings must be unit-normalized."""
    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_candidates(
    *,
    project: str,
    train_items: list[dict],
    math500_items: list[dict],
    ngram_hit_pairs: set[tuple[str, str]],
    encoder: Encoder,
    train_embs_cache: str | Path,
    math500_embs_cache: str | Path,
) -> list[dict]:
    """Run the embedding retrieval stage end-to-end.

    Returns candidates sorted by similarity score descending. Empty-text items
    are skipped; already-flagged n-gram pairs are excluded.
    """
    config = encoder.config

    valid_train = [(i, item) for i, item in enumerate(train_items) if item.get("problem", "").strip()]
    valid_math500 = [(i, item) for i, item in enumerate(math500_items) if item.get("problem", "").strip()]

    train_indices = [i for i, _ in valid_train]
    train_texts = [item["problem"] for _, item in valid_train]
    m500_indices = [i for i, _ in valid_math500]
    m500_texts = [item["problem"] for _, item in valid_math500]

    _log.info(
        "embedding retrieval: project=%s train=%d math500=%d (excluding %d c_lex pairs)",
        project, len(valid_train), len(valid_math500), len(ngram_hit_pairs),
    )

    train_embs = encoder.encode_chunked(train_texts, train_embs_cache)
    index = build_faiss_index(train_embs)
    m500_embs = encoder.encode_chunked(m500_texts, math500_embs_cache)

    similarities, neighbors = index.search(m500_embs, config.top_k)

    candidates: list[dict] = []
    for m_pos, (sims, nbrs) in enumerate(zip(similarities, neighbors)):
        math_item = math500_items[m500_indices[m_pos]]
        for sim, nbr_pos in zip(sims, nbrs):
            if sim < config.sim_threshold:
                continue
            train_item = train_items[train_indices[nbr_pos]]
            pair = (math_item["math500_id"], train_item["train_id"])
            if pair in ngram_hit_pairs:
                continue
            candidates.append({
                "project": project,
                "contamination_type": "c_sem_candidate",
                "math500_id": math_item["math500_id"],
                "math500_problem": math_item["problem"],
                "math500_answer": math_item.get("answer", ""),
                "math500_subject": math_item.get("subject", ""),
                "math500_level": math_item.get("level", -1),
                "train_id": train_item["train_id"],
                "train_problem": train_item.get("problem", ""),
                "train_solution": train_item.get("solution", "")[:500],
                "similarity_score": float(sim),
                "ngram_overlap": 0,
            })

    candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
    unique = len({c["math500_id"] for c in candidates})
    _log.info("  %d candidates above %.2f (%d unique MATH-500)",
              len(candidates), config.sim_threshold, unique)
    return candidates
