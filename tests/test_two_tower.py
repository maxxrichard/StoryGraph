import numpy as np
import pytest

from storygraph.config import TwoTowerConfig

pytestmark = [pytest.mark.ml, pytest.mark.torch]


def _data(n_users=64, n_titles=40, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    clusters = rng.normal(size=(4, dim))
    title_c = rng.integers(0, 4, n_titles)
    user_c = rng.integers(0, 4, n_users)
    title_x = clusters[title_c] + 0.2 * rng.normal(size=(n_titles, dim))
    user_x = clusters[user_c] + 0.2 * rng.normal(size=(n_users, dim))
    pairs = np.array(
        [
            [u, rng.choice(np.where(title_c == user_c[u])[0])]
            for u in range(n_users)
            for _ in range(4)
        ]
    )
    return user_x.astype(np.float32), title_x.astype(np.float32), pairs, user_c, title_c


def test_two_tower_trains_and_exports():
    torch = pytest.importorskip("torch")
    from storygraph.training.two_tower import train_two_tower

    user_x, title_x, pairs, user_c, title_c = _data()
    cfg = TwoTowerConfig(
        embedding_dim=16,
        content_dim=title_x.shape[1],
        hidden=[32],
        epochs=30,
        batch_size=64,
        learning_rate=5e-3,
        temperature=0.1,
        dropout=0.0,
    )
    model, losses = train_two_tower(user_x, title_x, pairs, cfg)
    assert losses[-1] < losses[0]
    weights = model.export()
    u = weights.encode_users(user_x)
    t = weights.encode_titles(title_x)
    assert np.allclose(np.linalg.norm(u, axis=1), 1, atol=1e-4)
    # numpy forward matches torch forward
    with torch.no_grad():
        ut = model.encode_users(torch.as_tensor(user_x)).numpy()
    assert np.allclose(u, ut, atol=1e-4)
    # retrieval quality: nearest title shares the user's cluster most of the time
    nearest = (u @ t.T).argmax(axis=1)
    assert (title_c[nearest] == user_c).mean() > 0.8


def test_hard_negative_loss_shape():
    torch = pytest.importorskip("torch")
    from storygraph.training.two_tower import TwoTower

    cfg = TwoTowerConfig(
        embedding_dim=8,
        content_dim=6,
        hidden=[8],
        hard_negatives_per_positive=3,
        negatives="mixed",
        dropout=0.0,
    )
    m = TwoTower(user_dim=5, cfg=cfg)
    loss = m.loss(torch.randn(4, 5), torch.randn(4, 6), torch.randn(4, 3, 6))
    assert loss.ndim == 0 and loss.item() > 0


def test_weights_roundtrip(tmp_path):
    pytest.importorskip("torch")
    from storygraph.training.two_tower import TwoTower, TwoTowerWeights

    cfg = TwoTowerConfig(embedding_dim=8, content_dim=6, hidden=[8], dropout=0.0)
    w = TwoTower(user_dim=5, cfg=cfg).export()
    w.save(str(tmp_path / "w.npy"))
    w2 = TwoTowerWeights.load(str(tmp_path / "w.npy"))
    x = np.random.default_rng(0).normal(size=(3, 5)).astype(np.float32)
    assert np.allclose(w.encode_users(x), w2.encode_users(x))
