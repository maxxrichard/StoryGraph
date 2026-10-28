import httpx
import pytest

from storygraph.serving.llm_client import CircuitBreaker, LLMUnavailable, StubLLM, VLLMClient


def _client(handler, retries=1):
    c = VLLMClient(base_url="http://vllm.test/v1", retries=retries, adapter="tagger")
    c._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=c.base_url)
    return c


def test_vllm_client_success_and_adapter_selection():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = req.read()
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"genres": []}'}}]})

    out = _client(handler).complete_json("sys", "user", {"type": "object"})
    assert out == '{"genres": []}'
    assert b'"model":"tagger"' in seen["body"] and b"guided_json" in seen["body"]


def test_vllm_client_retries_then_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503)

    with pytest.raises(LLMUnavailable):
        _client(handler, retries=2).complete("s", "u")
    assert calls["n"] == 3


def test_circuit_breaker_opens_and_half_opens(monkeypatch):
    cb = CircuitBreaker(failure_threshold=2, reset_after_s=10)
    assert cb.allow()
    cb.record(False)
    cb.record(False)
    assert not cb.allow()
    t = {"now": 0.0}
    monkeypatch.setattr("time.monotonic", lambda: t["now"])
    cb.opened_at = 0.0
    t["now"] = 11.0
    assert cb.allow()
    cb.record(True)
    assert cb.failures == 0 and cb.allow()


def test_stub_llm_failure_injection():
    s = StubLLM(fail_every=2)
    s.complete("a", "b")
    with pytest.raises(LLMUnavailable):
        s.complete("a", "b")
