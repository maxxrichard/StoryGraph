from datetime import date

import numpy as np

from storygraph.config import CandidateConfig
from storygraph.recsys.explain import llm_explanation_prompt, template_explanation
from storygraph.recsys.features import (
    CROSS_FEATURES,
    GENRE_INDEX,
    ListenerFeatures,
    TitleFeatures,
    cross_features,
    listener_from_history,
)
from storygraph.recsys.pipeline import Recommender
from storygraph.recsys.retrieval import InMemoryIndex, retrieve_candidates
from storygraph.training.ranker import LinearRanker


def _titles(n: int = 30, dim: int = 16, seed: int = 0) -> dict[str, TitleFeatures]:
    rng = np.random.default_rng(seed)
    out = {}
    genres = ["crime", "memoir", "economics", "horror"]
    for i in range(n):
        v = rng.normal(size=dim).astype(np.float32)
        v /= np.linalg.norm(v)
        out[f"t{i}"] = TitleFeatures(
            title_id=f"t{i}",
            content_vec=v,
            genres=[genres[i % 4]],
            themes=["trust"] if i % 3 == 0 else ["grief"],
            moods=["tense"],
            duration_min=100 + i,
            narrator=f"n{i % 5}",
            release_date=date(2026, 1, 1),
            popularity_7d=float(n - i),
        )
    return out


def _index(titles: dict[str, TitleFeatures]) -> InMemoryIndex:
    idx = InMemoryIndex()
    for t in titles.values():
        idx.add(t.title_id, {"content_vec": t.content_vec}, popularity=t.popularity_7d)
    return idx


def test_listener_from_history_builds_affinities():
    titles = _titles()
    hist = [
        {
            "title_id": "t0",
            "date": date(2026, 9, 1),
            "started_seconds": 100,
            "listened_seconds": 100,
            "finished": True,
        },
        {
            "title_id": "t1",
            "date": date(2026, 6, 1),
            "started_seconds": 100,
            "listened_seconds": 10,
            "finished": False,
        },
    ]
    lf = listener_from_history("u", hist, titles, as_of=date(2026, 9, 10))
    assert lf.history == ["t0", "t1"]
    assert lf.genre_affinity[GENRE_INDEX["crime"]] > lf.genre_affinity[GENRE_INDEX["memoir"]]
    assert np.isclose(lf.genre_affinity.sum(), 1.0)
    assert lf.finish_rate_90d == 1.0  # t1 is older than 90 days
    assert "n0" in lf.recent_narrators and lf.vector().shape == (lf.dim,)


def test_cross_features_shape_and_matches():
    titles = _titles()
    lf = ListenerFeatures(listener_id="u", recent_narrators={"n1"}, recent_themes={"trust"})
    lf.genre_affinity[GENRE_INDEX["memoir"]] = 1.0
    x = cross_features(lf, titles["t1"], 0.5, 0.4, 0.3, 2, as_of=date(2026, 2, 1))
    f = dict(zip(CROSS_FEATURES, x.tolist()))
    assert x.shape == (len(CROSS_FEATURES),)
    assert (
        f["genre_match"] == 1.0
        and f["same_narrator"] == 1.0
        and f["two_tower_score"] == 0.5
        and f["n_sources"] == 2
    )
    assert f["theme_overlap"] == 0.0  # t1 has grief


def test_retrieve_candidates_unions_sources():
    titles = _titles()
    idx = _index(titles)
    q = titles["t3"].content_vec
    cfg = CandidateConfig(k_two_tower=0, k_content_knn=5, k_popular=3, k_total_max=100)
    cands = retrieve_candidates(idx, None, q, exclude={"t3"}, cfg=cfg)
    ids = {c.title_id for c in cands}
    assert "t3" not in ids
    assert {"t0", "t1", "t2"} <= ids  # most popular
    assert any("content_knn" in c.sources for c in cands) and any(
        "popular" in c.sources for c in cands
    )
    best = [max(c.scores.values()) for c in cands]
    assert best == sorted(best, reverse=True)


def test_recommender_end_to_end():
    titles = _titles()
    idx = _index(titles)
    w = np.zeros(len(CROSS_FEATURES))
    w[CROSS_FEATURES.index("content_knn_score")] = 1.0
    w[CROSS_FEATURES.index("genre_match")] = 0.5
    ranker = LinearRanker(
        weights=w, feature_names=list(CROSS_FEATURES), mean=np.zeros(len(w)), std=np.ones(len(w))
    )
    rec = Recommender(
        index=idx,
        titles=titles,
        ranker=ranker,
        candidates_cfg=CandidateConfig(k_content_knn=10, k_popular=5),
    )
    lf = listener_from_history(
        "u",
        [
            {
                "title_id": "t4",
                "date": date(2026, 9, 1),
                "started_seconds": 10,
                "listened_seconds": 10,
            }
        ],
        titles,
        as_of=date(2026, 9, 2),
    )
    resp = rec.recommend(lf, k=5)
    assert len(resp.items) == 5
    assert [i.rank for i in resp.items] == [1, 2, 3, 4, 5]
    assert all(i.title_id != "t4" for i in resp.items)
    assert resp.items[0].score >= resp.items[-1].score
    assert all(i.explanation for i in resp.items)


def test_recommender_handles_cold_start():
    titles = _titles()
    rec = Recommender(
        index=_index(titles),
        titles=titles,
        ranker=LinearRanker(
            np.ones(len(CROSS_FEATURES)),
            list(CROSS_FEATURES),
            np.zeros(len(CROSS_FEATURES)),
            np.ones(len(CROSS_FEATURES)),
        ),
    )
    resp = rec.recommend(ListenerFeatures(listener_id="new"), k=3)
    assert len(resp.items) == 3 and all("popular" in i.retrieval_sources for i in resp.items)


def test_explanations():
    titles = _titles()
    lf = ListenerFeatures(listener_id="u", recent_narrators={"n1"}, recent_themes={"grief"})
    lf.genre_affinity[GENRE_INDEX["memoir"]] = 0.9
    text = template_explanation(lf, titles["t1"], ["two_tower"])
    assert "memoir" in text and text.endswith(".")
    assert template_explanation(
        ListenerFeatures(listener_id="x"), titles["t1"], ["popular"]
    ).startswith("Popular")
    prompt = llm_explanation_prompt(lf, titles["t1"])
    assert "No spoilers" in prompt and "memoir" in prompt
