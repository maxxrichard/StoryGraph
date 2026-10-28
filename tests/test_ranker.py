import numpy as np
import pytest

from storygraph.config import RankerConfig
from storygraph.evaluation.ranking_metrics import (
    evaluate_rankings,
    mrr,
    ndcg_at_k,
    recall_at_k,
    relative_lift,
)
from storygraph.training.ranker import LinearRanker, relevance_grade


def _synthetic(n_groups=40, per_group=12, dim=5, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_groups * per_group, dim))
    true_w = np.array([2.0, -1.0, 0.5, 0.0, 0.0])
    score = X @ true_w + rng.normal(scale=0.3, size=len(X))
    y = np.zeros(len(X), dtype=int)
    for g in range(n_groups):
        sl = slice(g * per_group, (g + 1) * per_group)
        order = np.argsort(-score[sl])
        y[sl][order[:1]] = 3
        y[sl][order[1:3]] = 2
        y[sl][order[3:6]] = 1
    return X, y, np.full(n_groups, per_group)


def _ndcg(model, X, y, group, k=10):
    out, off = [], 0
    for g in group:
        sl = slice(off, off + g)
        ids = [str(i) for i in range(g)]
        ranked = [ids[i] for i in np.argsort(-model.predict(X[sl]))]
        out.append(ndcg_at_k(ranked, dict(zip(ids, y[sl].astype(float))), k))
        off += g
    return float(np.mean(out))


def test_linear_ranker_learns_ordering():
    X, y, group = _synthetic()
    model = LinearRanker.fit(X, y, group, [f"f{i}" for i in range(X.shape[1])], epochs=20)
    assert _ndcg(model, X, y, group) > 0.85
    assert model.weights[0] > 0 > model.weights[1]


def test_linear_ranker_roundtrip(tmp_path):
    X, y, group = _synthetic(n_groups=5)
    m = LinearRanker.fit(X, y, group, ["a", "b", "c", "d", "e"], epochs=3)
    m.save(tmp_path / "r.json")
    m2 = LinearRanker.load(tmp_path / "r.json")
    assert np.allclose(m.predict(X), m2.predict(X))


def test_relevance_grade():
    assert relevance_grade(10, 100, True, False) == 3
    assert relevance_grade(70, 100, False, False) == 2
    assert relevance_grade(20, 100, False, False) == 1
    assert relevance_grade(1, 100, False, False) == 0
    assert relevance_grade(0, 0, False, False) == 0


def test_ranking_metrics():
    rel = {"a": 3.0, "b": 1.0, "c": 0.0}
    assert ndcg_at_k(["a", "b", "c"], rel, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["c", "b", "a"], rel, 10) < 0.8
    assert recall_at_k(["c", "a"], {"a", "b"}, 2) == 0.5
    assert mrr(["c", "b"], {"b"}) == 0.5 and mrr(["c"], {"b"}) == 0.0
    res = evaluate_rankings({"u": ["a", "b", "c"]}, {"u": rel}, ks=(10,), recall_k=50)
    assert res["ndcg@10"] == pytest.approx(1.0) and res["recall@50"] == 1.0 and res["mrr"] == 1.0
    lift = relative_lift({"ndcg@10": 0.57}, {"ndcg@10": 0.50})
    assert lift["ndcg@10"] == pytest.approx(0.14)


@pytest.mark.ml
@pytest.mark.lightgbm
def test_lightgbm_ranker_beats_linear():
    pytest.importorskip("lightgbm")
    from storygraph.training.ranker import LightGBMRanker

    X, y, group = _synthetic(n_groups=120, per_group=15)
    cfg = RankerConfig(n_estimators=120, num_leaves=7, min_data_in_leaf=5, learning_rate=0.1)
    names = [f"f{i}" for i in range(X.shape[1])]
    lgbm = LightGBMRanker(cfg, names).fit(X, y, group)
    assert _ndcg(lgbm, X, y, group) > 0.9
    imp = lgbm.feature_importance()
    assert max(imp, key=imp.get) in {"f0", "f1"}
