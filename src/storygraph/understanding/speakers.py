"""Speaker signals: aggregate diarization output into per-title speaker profiles.

Diarization itself (pyannote) runs in the ASR job; this module turns ``speaker`` labels on
utterances into shares and heuristic roles (narrator vs host vs guest vs character).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from storygraph.schemas import Kind, SpeakerProfile, Utterance


def speaker_shares(utterances: Sequence[Utterance]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for u in utterances:
        counts[u.speaker or "unknown"] += len(u.text.split())
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def infer_roles(shares: dict[str, float], kind: Kind) -> list[SpeakerProfile]:
    """Heuristic roles: dominant voice is narrator (audiobook) or host (podcast)."""
    ordered = sorted(shares.items(), key=lambda kv: -kv[1])
    out: list[SpeakerProfile] = []
    for i, (spk, share) in enumerate(ordered):
        if kind == Kind.AUDIOBOOK:
            role = "narrator" if i == 0 and share >= 0.5 else "character"
        elif kind == Kind.PODCAST:
            role = "host" if i == 0 or share >= 0.35 else "guest"
        else:
            role = "unknown"
        out.append(SpeakerProfile(speaker_id=spk, share=share, role=role))  # type: ignore[arg-type]
    return out


def prune(profiles: Sequence[SpeakerProfile], min_share: float = 0.02) -> list[SpeakerProfile]:
    return [p for p in profiles if p.share >= min_share]
