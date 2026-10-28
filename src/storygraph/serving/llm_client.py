"""Client for a vLLM OpenAI-compatible endpoint with LoRA adapter selection, guided JSON
decoding, timeouts, bounded retries and a circuit breaker.

Implements both ``LLMBackend`` (tagger) and ``TextLLM`` (summariser / judge).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    reset_after_s: float = 30.0
    failures: int = 0
    opened_at: float | None = None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at >= self.reset_after_s:
            self.opened_at = None  # half-open: let one request through
            self.failures = 0
            return True
        return False

    def record(self, ok: bool) -> None:
        if ok:
            self.failures = 0
            self.opened_at = None
            return
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.monotonic()


@dataclass
class VLLMClient:
    base_url: str = "http://localhost:8000/v1"
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    adapter: str | None = None
    timeout_s: float = 20.0
    retries: int = 2
    temperature: float = 0.0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    _client: httpx.Client | None = None

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout_s)
        return self._client

    def _chat(
        self, messages: list[dict[str, str]], max_tokens: int, extra: dict[str, Any] | None = None
    ) -> str:
        if not self.breaker.allow():
            raise LLMUnavailable("circuit open")
        payload: dict[str, Any] = {
            "model": self.adapter or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        if extra:
            payload.update(extra)
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self._http().post("/chat/completions", json=payload)
                r.raise_for_status()
                self.breaker.record(True)
                return r.json()["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last = exc
                self.breaker.record(False)
                log.warning(
                    "vLLM call failed (attempt %d/%d): %s", attempt + 1, self.retries + 1, exc
                )
                if attempt < self.retries:
                    time.sleep(0.2 * (2**attempt))
        raise LLMUnavailable(f"vLLM call failed after {self.retries + 1} attempts") from last

    # LLMBackend protocol (tagger)
    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._chat(
            messages,
            max_tokens=512,
            extra={"guided_json": schema, "response_format": {"type": "json_object"}},
        )

    # TextLLM protocol (summariser, judge)
    def complete(self, system: str, user: str, max_tokens: int = 256) -> str:
        return self._chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
        )


class StubLLM:
    """Deterministic stand-in used by tests and the local demo.

    Produces taxonomy-conformant tags from simple keyword heuristics and template summaries, so the
    full pipeline can be exercised end-to-end without a model server.
    """

    def __init__(self, fail_every: int = 0):
        self.calls = 0
        self.fail_every = fail_every

    def _maybe_fail(self) -> None:
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            raise LLMUnavailable("stub failure")

    @staticmethod
    def _has(text: str, *words: str) -> bool:
        return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        self._maybe_fail()
        # only look at the transcript body, not the taxonomy block in the prompt header
        text = user.split("TRANSCRIPT:", 1)[-1].lower()
        genres, themes, moods = [], [], []
        if self._has(text, "bank", "deposits", "regulator", "treasury"):
            genres.append({"name": "economics", "confidence": 0.9})
            themes += [
                {"name": "coordination_failure", "confidence": 0.85},
                {"name": "trust", "confidence": 0.7},
                {"name": "institutions", "confidence": 0.6},
            ]
            moods += [
                {"name": "analytical", "confidence": 0.8},
                {"name": "tense", "confidence": 0.6},
            ]
        if self._has(text, "lighthouse", "fog", "boat", "sea", "harbourmaster"):
            genres += [
                {"name": "literary fiction", "confidence": 0.8},
                {"name": "ghost story", "confidence": 0.6},
            ]
            themes += [
                {"name": "grief", "confidence": 0.8},
                {"name": "isolation", "confidence": 0.75},
                {"name": "the sea", "confidence": 0.7},
            ]
            moods += [
                {"name": "eerie", "confidence": 0.85},
                {"name": "melancholic", "confidence": 0.7},
            ]
        if self._has(text, "steppe", "herd", "herders", "pasture", "horse", "mare"):
            genres += [
                {"name": "memoir", "confidence": 0.85},
                {"name": "travel", "confidence": 0.7},
            ]
            themes += [
                {"name": "community", "confidence": 0.8},
                {"name": "language", "confidence": 0.6},
                {"name": "trust", "confidence": 0.6},
            ]
            moods += [
                {"name": "contemplative", "confidence": 0.8},
                {"name": "warm", "confidence": 0.7},
            ]
        if not genres:
            genres.append({"name": "literary fiction", "confidence": 0.5})
        tension = (
            0.7
            if any(w in text for w in ("footsteps", "panic", "frightening", "under again"))
            else 0.3
        )
        return json.dumps(
            {
                "genres": genres,
                "themes": themes,
                "moods": moods,
                "narrative_arc": None,
                "tension": tension,
                "content_warnings": ["death"] if self._has(text, "dead", "died") else [],
                "speakers": [],
                "one_line_summary": "A short, spoiler-free summary.",
            }
        )

    def complete(self, system: str, user: str, max_tokens: int = 256) -> str:
        self._maybe_fail()
        if "CANDIDATE SUMMARY" in user:
            return json.dumps(
                {
                    "faithfulness": 4,
                    "spoiler_safety": 5,
                    "coverage": 4,
                    "fluency": 4,
                    "rationale": "stub",
                }
            )
        if "Write ONE sentence" in user:
            return "A quiet, atmospheric listen that matches the moods you keep coming back to."
        return json.dumps(
            {
                "heading": "Fog on the landing",
                "summary": "The keeper notes an unfamiliar boat and an uneasy calm settles over the point. A visit from the mainland raises a question she is not ready to ask.",
            }
        )
