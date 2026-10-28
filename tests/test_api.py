import pytest
from fastapi.testclient import TestClient

from storygraph.serving.api import build_demo_context, create_app


@pytest.fixture(scope="module")
def client(request):
    import os

    root = os.path.join(os.path.dirname(__file__), "..")
    cwd = os.getcwd()
    os.chdir(root)
    try:
        ctx = build_demo_context()
    finally:
        os.chdir(cwd)
    return TestClient(create_app(ctx))


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["titles"] == 3


def test_recommend(client):
    body = {
        "listener_id": "u-1",
        "history": [{"title_id": "t-0001", "started_seconds": 3600, "listened_seconds": 3300}],
        "k": 5,
    }
    r = client.post("/v1/recommend", json=body)
    assert r.status_code == 200
    data = r.json()
    ids = [i["title_id"] for i in data["items"]]
    assert "t-0001" not in ids and len(ids) == 2
    assert data["variant"] in {"control", "treatment"} and data["latency_ms"] >= 0
    assert all(i["explanation"] for i in data["items"])
    # variant override is honoured
    r = client.post("/v1/recommend", json={**body, "variant_override": "control"})
    assert r.json()["variant"] == "control" and r.json()["model_version"] == "item_cf_v12"


def test_recommend_validation(client):
    assert client.post("/v1/recommend", json={"listener_id": "u", "k": 0}).status_code == 422


def test_signals_and_chapters(client):
    r = client.get("/v1/titles/t-0002/signals")
    assert r.status_code == 200 and "economics" in [g["name"] for g in r.json()["genres"]]
    assert client.get("/v1/titles/nope/signals").status_code == 404
    r = client.get("/v1/titles/t-0001/chapters?limit=2")
    assert r.status_code == 200 and 1 <= len(r.json()) <= 2
    assert all(c["spoiler_free"] for c in r.json())


def test_experiment_endpoint(client):
    r = client.get("/v1/experiment/listener-42")
    assert r.status_code == 200 and r.json()["variant"] in {"control", "treatment"}
    assert client.get("/healthz").json()["availability"] == 1.0
