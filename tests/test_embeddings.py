import numpy as np
import pytest

from storygraph.understanding.embeddings import HashingEmbedder, get_embedder, pool_title_embedding


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(dim=64)
    a = e.encode(["the quick brown fox", "the quick brown fox"])
    assert np.allclose(a[0], a[1])
    assert np.isclose(np.linalg.norm(a[0]), 1.0)


def test_similar_texts_are_closer():
    e = HashingEmbedder(dim=256)
    v = e.encode(
        [
            "the keeper watched the sea and fog",
            "the keeper watched the fog and sea",
            "deposits capital bank regulator memo",
        ]
    )
    assert v[0] @ v[1] > v[0] @ v[2]


def test_pool_title_embedding_weights():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    v = pool_title_embedding(vecs, [3, 1])
    assert v[0] > v[1] and np.isclose(np.linalg.norm(v), 1.0)
    with pytest.raises(ValueError):
        pool_title_embedding(np.zeros((0, 2)))


def test_get_embedder_hashing():
    assert get_embedder("hashing", dim=32).dim == 32
