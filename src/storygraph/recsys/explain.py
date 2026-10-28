"""Explained picks: short, honest reasons a title was recommended.

Two tiers:

* :func:`template_explanation` – deterministic, from the cross-features (always available,
  used when the LLM budget or latency SLO is tight).
* :func:`llm_explanation_prompt` – a prompt for the summariser LoRA to write a one-sentence
  reason grounded only in the provided signals (never in hallucinated plot details).
"""

from __future__ import annotations

from storygraph.recsys.features import ListenerFeatures, TitleFeatures


def _humanise(label: str) -> str:
    return label.replace("_", " ")


def template_explanation(
    listener: ListenerFeatures, title: TitleFeatures, sources: list[str]
) -> str:
    reasons: list[str] = []
    from storygraph.recsys.features import GENRE_INDEX, MOOD_INDEX

    top_genres = sorted(
        ((listener.genre_affinity[GENRE_INDEX[g]], g) for g in title.genres if g in GENRE_INDEX),
        reverse=True,
    )
    if top_genres and top_genres[0][0] > 0.15:
        reasons.append(f"you listen to a lot of {_humanise(top_genres[0][1])}")
    shared_themes = [t for t in title.themes if t in listener.recent_themes][:2]
    if shared_themes:
        reasons.append(
            "it explores "
            + " and ".join(_humanise(t) for t in shared_themes)
            + ", like recent titles you finished"
        )
    top_moods = sorted(
        ((listener.mood_affinity[MOOD_INDEX[m]], m) for m in title.moods if m in MOOD_INDEX),
        reverse=True,
    )
    if top_moods and top_moods[0][0] > 0.2:
        reasons.append(f"the mood is {_humanise(top_moods[0][1])}")
    if title.narrator and title.narrator in listener.recent_narrators:
        reasons.append("it's read by a narrator you've enjoyed")
    if not reasons:
        if "popular" in sources:
            return "Popular with listeners this week."
        return "Similar to titles you've been listening to."
    text = "; ".join(reasons[:2])
    return text[0].upper() + text[1:] + "."


def llm_explanation_prompt(
    listener: ListenerFeatures, title: TitleFeatures, chapter_hook: str | None = None
) -> str:
    from storygraph.recsys.features import GENRES, MOODS

    fav_genres = [
        GENRES[i]
        for i in listener.genre_affinity.argsort()[::-1][:3]
        if listener.genre_affinity[i] > 0
    ]
    fav_moods = [
        MOODS[i]
        for i in listener.mood_affinity.argsort()[::-1][:2]
        if listener.mood_affinity[i] > 0
    ]
    return (
        "Write ONE sentence (max 25 words) telling a listener why they might like this title. "
        "Use only the facts below. No spoilers, no superlatives, no invented plot details.\n\n"
        f"Listener likes genres: {', '.join(fav_genres) or 'unknown'}; moods: {', '.join(fav_moods) or 'unknown'}; "
        f"recent themes: {', '.join(sorted(listener.recent_themes)[:5]) or 'unknown'}.\n"
        f"Title genres: {', '.join(title.genres)}; themes: {', '.join(title.themes)}; moods: {', '.join(title.moods)}; "
        f"narrative arc: {title.narrative_arc or 'unknown'}; duration: {title.duration_min:.0f} min.\n"
        + (f"Opening hook: {chapter_hook}\n" if chapter_hook else "")
    )
