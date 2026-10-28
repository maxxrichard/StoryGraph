"""Prompt templates. Kept in one place so prompt changes are reviewable diffs."""

from __future__ import annotations

SUMMARY_SYSTEM = (
    "You write short, spoiler-free chapter summaries for an audio app. Summaries help a listener "
    "decide whether to start or resume, so describe setting, who is involved and what question the "
    "chapter raises, without revealing how it resolves. Plain language, no marketing tone."
)

SUMMARY_USER = """\
TITLE: {title}
KIND: {kind}
CHAPTER {index} of {total} ({minutes:.0f} min)
GENRES: {genres}
MOODS: {moods}

TRANSCRIPT:
{text}

Return JSON: {{"heading": "<<= 8 words>", "summary": "<2-4 sentences, <= 90 words, spoiler-free>"}}"""

JUDGE_SYSTEM = (
    "You are a strict editorial reviewer. Score the candidate chapter summary against the transcript "
    "on each criterion from 1 (poor) to 5 (excellent). Be calibrated: a 5 is rare. Return JSON only."
)

JUDGE_USER = """\
TRANSCRIPT (excerpt):
{text}

CANDIDATE SUMMARY:
{summary}

Criteria:
- faithfulness: every claim is supported by the transcript (hallucination => 1)
- spoiler_safety: does not reveal resolutions, twists, or endings
- coverage: captures the chapter's setting, people, and central question
- fluency: clear, concise, natural prose

Return JSON: {{"faithfulness": n, "spoiler_safety": n, "coverage": n, "fluency": n, "rationale": "<one sentence>"}}"""
