"""Embedding retrieval, with the encoder replaced where a model would be needed."""

import numpy as np
import pytest

from contamination_audit.filters.embedding import Encoder, nearest


def test_nearest_finds_each_items_best_document_across_chunks():
    corpus = np.eye(3, dtype=np.float32)
    items = np.array([[0, 1, 0], [0.6, 0.8, 0]], dtype=np.float32)
    found = nearest(corpus, items, {"first": [0], "second": [1]}, chunk=2)
    assert found.item_document.tolist() == [1, 1]
    assert np.allclose(found.item_similarity, [1.0, 0.8])
    assert np.allclose(found.document_similarity["first"], [0, 1, 0])
    assert np.allclose(found.document_similarity["second"], [0.6, 0.8, 0])


def test_corpus_embeddings_are_computed_once_and_read_back(tmp_path, monkeypatch):
    calls = []

    def encode(self, texts):
        calls.append(len(texts))
        return np.full((len(texts), 4), 0.5, dtype=np.float32)

    monkeypatch.setattr(Encoder, "encode", encode)
    first = Encoder().encode_cached(["a", "b"], tmp_path, "corpus")
    second = Encoder().encode_cached(["a", "b"], tmp_path, "corpus")
    assert calls == [2]
    assert np.array_equal(first, second)
    # Different texts are a different cache entry.
    Encoder().encode_cached(["a", "c"], tmp_path, "corpus")
    assert calls == [2, 2]


def test_a_crash_keeps_the_finished_shards(tmp_path, monkeypatch):
    calls = []

    def encode(self, texts):
        calls.append(list(texts))
        if texts[0] == "c" and len(calls) == 2:
            raise RuntimeError("CUDA error")
        return np.array([[ord(t), 0.0] for t in texts], dtype=np.float32)

    monkeypatch.setattr(Encoder, "encode", encode)
    encoder = Encoder(shard_size=2)
    with pytest.raises(RuntimeError):
        encoder.encode_cached(["a", "b", "c"], tmp_path, "corpus")

    vectors = encoder.encode_cached(["a", "b", "c"], tmp_path, "corpus")
    assert calls == [["a", "b"], ["c"], ["c"]]
    assert vectors[:, 0].tolist() == [ord("a"), ord("b"), ord("c")]
