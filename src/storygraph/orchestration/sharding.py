"""Catalog sharding and idempotency bookkeeping shared by every orchestrator."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from storygraph.schemas import Title


def shard_of(title_id: str, shard_count: int) -> int:
    return int(hashlib.md5(title_id.encode()).hexdigest(), 16) % shard_count  # noqa: S324 - not security


def plan_shards(
    titles: Iterable[Title], shard_count: int, done_keys: set[str] | None = None
) -> dict[int, list[Title]]:
    """Group titles into shards, skipping titles whose content_key is already processed."""
    done = done_keys or set()
    shards: dict[int, list[Title]] = {}
    for t in titles:
        if t.content_key in done:
            continue
        shards.setdefault(shard_of(t.title_id, shard_count), []).append(t)
    return shards
