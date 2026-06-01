"""FAISS index + retrieval shape sanity check.

Avoids loading sentence-transformers (heavy + downloads a model on first use).
Instead we feed hand-crafted unit-normalized vectors directly into the FAISS
index and verify the top-K retrieval shape and ordering.
"""

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from contamination_audit.embedding import build_faiss_index  # noqa: E402


def _unit(*components: float) -> np.ndarray:
    v = np.array(components, dtype="float32")
    return v / np.linalg.norm(v)


def test_faiss_top_k_returns_expected_shape():
    train_embs = np.stack([
        _unit(1, 0, 0),
        _unit(0, 1, 0),
        _unit(0, 0, 1),
        _unit(1, 1, 0),
    ])
    index = build_faiss_index(train_embs)
    query_embs = np.stack([_unit(1, 0, 0), _unit(0, 1, 0)])
    sims, neighbors = index.search(query_embs, k=2)
    assert sims.shape == (2, 2)
    assert neighbors.shape == (2, 2)


def test_faiss_top_neighbor_is_correct():
    """Query vector aligned with item 0 must return item 0 as the top neighbor."""
    train_embs = np.stack([
        _unit(1, 0, 0),
        _unit(0, 1, 0),
        _unit(0, 0, 1),
    ])
    index = build_faiss_index(train_embs)
    sims, neighbors = index.search(np.stack([_unit(1, 0, 0)]), k=3)
    assert neighbors[0, 0] == 0
    assert sims[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_faiss_index_is_deterministic():
    """Two index builds on identical inputs must yield identical search output."""
    rng = np.random.default_rng(0)
    train_embs = rng.standard_normal((10, 8)).astype("float32")
    train_embs /= np.linalg.norm(train_embs, axis=1, keepdims=True)
    query = train_embs[:3].copy()

    a = build_faiss_index(train_embs).search(query, k=5)
    b = build_faiss_index(train_embs).search(query, k=5)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
