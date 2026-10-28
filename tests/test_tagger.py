import json

import pytest

from storygraph.config import AggregationConfig
from storygraph.schemas import ContentSignals, Label, NarrativeArc, Segment
from storygraph.serving.llm_client import StubLLM
from storygraph.understanding.tagger import (
    SegmentTagger,
    aggregate_title_signals,
    build_prompt,
    parse_signals,
)
from storygraph.understanding.taxonomy import DEFAULT_TAXONOMY, Taxonomy


def _seg(i: int, text: str, words: int = 100) -> Segment:
    return Segment(
        title_id="t", index=i, start=i * 10.0, end=i * 10.0 + 10, text=text, word_count=words
    )


def test_taxonomy_snap_handles_synonyms_and_variants():
    tx = Taxonomy()
    assert tx.snap("Sci-Fi", "genres") == "science_fiction"
    assert tx.snap("literary fiction", "genres") == "literary_fiction"
    assert tx.snap("Ghost story", "genres") == "horror"
    assert tx.snap("the sea", "themes") == "the_sea"
    assert tx.snap("crime drama", "genres") == "crime"
    assert tx.snap("spooky", "moods") == "eerie"
    assert tx.snap("underwater basket weaving", "genres") is None


def test_parse_signals_tolerates_prose_and_off_taxonomy_labels():
    raw = 'Sure! Here is the JSON:\n{"genres": [{"name": "Sci-Fi", "confidence": 0.9}, {"name": "zzz", "confidence": 0.9}], "themes": ["grief"], "moods": [], "narrative_arc": "not_an_arc", "tension": 0.4}'
    s = parse_signals(raw, DEFAULT_TAXONOMY, "t", 0)
    assert s.label_names("genres") == ["science_fiction"]
    assert s.label_names("themes") == ["grief"]
    assert s.narrative_arc is None and s.tension == 0.4


def test_parse_signals_rejects_non_json():
    with pytest.raises(ValueError):
        parse_signals("no json here", DEFAULT_TAXONOMY, "t", 0)


def test_prompt_truncates_long_segments():
    seg = _seg(0, "word " * 10_000, 10_000)
    p = build_prompt(seg, DEFAULT_TAXONOMY, max_chars=500)
    assert "…" in p and len(p) < 2000
    assert "GENRES:" in p and "NARRATIVE_ARCS:" in p


def test_stub_tagger_end_to_end():
    tagger = SegmentTagger(StubLLM())
    sig = tagger.tag(
        _seg(0, "The lighthouse keeper watched the fog and the empty boat on the sea.")
    )
    assert "literary_fiction" in sig.label_names("genres")
    assert "eerie" in sig.label_names("moods")
    assert sig.segment_index == 0


def test_aggregate_title_signals():
    segs = [_seg(0, "a", 300), _seg(1, "b", 100), _seg(2, "c", 100)]
    per = [
        ContentSignals(
            title_id="t",
            segment_index=0,
            genres=[Label(name="crime", confidence=0.9)],
            themes=[Label(name="trust", confidence=0.8), Label(name="grief", confidence=0.5)],
            moods=[Label(name="tense", confidence=0.9)],
            narrative_arc=NarrativeArc.MAN_IN_HOLE,
            tension=0.8,
            content_warnings=["violence"],
        ),
        ContentSignals(
            title_id="t",
            segment_index=1,
            genres=[Label(name="romance", confidence=0.9)],
            themes=[Label(name="trust", confidence=0.6)],
            moods=[Label(name="warm", confidence=0.7)],
            tension=0.2,
        ),
        ContentSignals(
            title_id="t",
            segment_index=2,
            genres=[Label(name="crime", confidence=0.6)],
            themes=[Label(name="money", confidence=0.9)],
            moods=[Label(name="tense", confidence=0.5)],
            narrative_arc=NarrativeArc.MAN_IN_HOLE,
            tension=0.5,
        ),
    ]
    agg = aggregate_title_signals(segs, per, AggregationConfig(themes_min_share=0.25))
    assert agg.label_names("genres")[0] == "crime"
    assert "trust" in agg.label_names("themes") and "money" not in agg.label_names(
        "themes"
    )  # 100/500 < 0.25
    assert agg.label_names("moods") == ["tense", "warm"]  # trajectory order
    assert agg.narrative_arc == NarrativeArc.MAN_IN_HOLE
    assert agg.content_warnings == ["violence"]
    assert 0.5 < agg.tension < 0.8  # type: ignore[operator]
    assert agg.segment_index is None


def test_aggregate_requires_alignment():
    with pytest.raises(ValueError):
        aggregate_title_signals([_seg(0, "a")], [])


def test_stub_json_validates_against_schema():
    raw = StubLLM().complete_json("s", "lighthouse fog", {})
    assert isinstance(json.loads(raw)["genres"], list)
