"""Embedding retrieval, the first stage of the LLM Decontaminator.

Yang et al. (2023) embed every training and test text with a sentence encoder,
retrieve each test item's most similar training texts by cosine similarity, and
ask an LLM whether each retrieved pair is the same problem. Their released code
(lm-sys/llm-decontaminator, vector_db.py) uses multi-qa-MiniLM-L6-cos-v1 and
shows top_k=1 in its example. This module reproduces the retrieval with the
same encoder; the experiment replaces the LLM's verdict with a threshold on the
cosine similarity, which it sweeps like the other filters' settings.

Encoding needs the optional `embedding` dependencies (sentence-transformers and
torch). It runs on a GPU when one is available, and the corpus embeddings are
cached under work/embeddings/.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
MODEL_REVISION = "b207367332321f8e44f96e224ef15bc607f4dbf0"


@dataclass(frozen=True)
class Encoder:
    model: str = MODEL
    revision: str = MODEL_REVISION
    # Small enough to keep each GPU job short: at 256, hours of encoding crashed
    # a laptop twice with a driver timeout (bugcheck 0x116, VIDEO_TDR_FAILURE).
    batch_size: int = 64
    shard_size: int = 100_000

    @cached_property
    def _encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                'embedding needs its optional dependencies: pip install -e ".[embedding]"'
            ) from error
        return SentenceTransformer(self.model, revision=self.revision)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-length embeddings, one row per text, so a dot product is a cosine."""
        vectors = self._encoder.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10_000,
        )
        return vectors.astype(np.float32)

    def encode_cached(self, texts: Sequence[str], directory: Path, name: str) -> np.ndarray:
        """`encode`, saved shard by shard under a name tied to the model and the texts.

        All of Dolci takes hours, so a crash must not cost the shards already
        done: running again picks up at the first shard that is missing.
        """
        digest = hashlib.sha256(f"{self.model}@{self.revision}\n".encode())
        for text in texts:
            digest.update(text.encode())
            digest.update(b"\0")
        shards = directory / f"{name}-{digest.hexdigest()[:16]}"
        shards.mkdir(parents=True, exist_ok=True)

        parts = []
        for number, start in enumerate(range(0, len(texts), self.shard_size)):
            path = shards / f"{number:05d}.npy"
            if not path.exists():
                log.info("embedding shard %d, from text %s", number, f"{start:,}")
                vectors = self.encode(texts[start : start + self.shard_size])
                # Half precision halves the cache; cosines move by about 1e-3 at most.
                # Written whole and then renamed, so a crash leaves no partial shard.
                partial = path.with_suffix(".partial.npy")
                np.save(partial, vectors.astype(np.float16))
                partial.replace(path)
            # Always the saved copy, so a fresh run and a resumed one compute
            # from exactly the same numbers.
            parts.append(np.load(path))
        return np.concatenate(parts).astype(np.float32)


@dataclass(frozen=True)
class Nearest:
    """For each item its most similar document, and for each document its most
    similar item within each group of items."""

    item_similarity: np.ndarray  # (items,) best cosine over the corpus
    item_document: np.ndarray  # (items,) index of that document
    document_similarity: dict[str, np.ndarray]  # group -> (documents,) best cosine


def nearest(
    corpus: np.ndarray,
    items: np.ndarray,
    groups: dict[str, Sequence[int]],
    chunk: int = 100_000,
) -> Nearest:
    """Top-1 retrieval of every item against every document, in chunks."""
    best = np.full(len(items), -np.inf, dtype=np.float32)
    where = np.zeros(len(items), dtype=np.int64)
    per_document = {group: np.empty(len(corpus), dtype=np.float32) for group in groups}
    for start in range(0, len(corpus), chunk):
        similarity = corpus[start : start + chunk] @ items.T  # (chunk, items)
        top = similarity.argmax(axis=0)
        values = similarity[top, np.arange(len(items))]
        better = values > best
        best[better] = values[better]
        where[better] = start + top[better]
        for group, members in groups.items():
            per_document[group][start : start + chunk] = similarity[:, list(members)].max(axis=1)
    return Nearest(best, where, per_document)
