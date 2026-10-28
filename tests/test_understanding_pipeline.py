import json

from storygraph.schemas import NarrativeArc
from storygraph.understanding.pipeline import understand_title


def test_understand_sample_titles(
    sample_titles, sample_transcripts, embedder, tagger, small_seg_cfg
):
    seen_genres = set()
    for t in sample_titles:
        u = understand_title(t, sample_transcripts[t.title_id], embedder, tagger, small_seg_cfg)
        assert len(u.segments) >= 1
        assert len(u.segment_signals) == len(u.segments)
        assert u.segment_embeddings.shape == (len(u.segments), embedder.dim)
        assert u.title_embedding.shape == (embedder.dim,)
        assert u.tension.shape == (12,)
        assert u.title_signals.narrative_arc in set(NarrativeArc)
        assert u.title_signals.speakers, "speaker fallback should populate"
        rec = u.to_record()
        json.dumps(rec)  # serialisable
        assert rec["segment_count"] == len(u.segments) and len(rec["content_vec"]) == embedder.dim
        seen_genres.update(rec["genres"])
    assert {"economics", "memoir"} <= seen_genres


def test_content_vectors_discriminate(
    sample_titles, sample_transcripts, embedder, tagger, small_seg_cfg
):
    vecs = {
        t.title_id: understand_title(
            t, sample_transcripts[t.title_id], embedder, tagger, small_seg_cfg
        ).title_embedding
        for t in sample_titles
    }
    # podcast about a bank run should be furthest from the lighthouse novel
    sim = lambda a, b: float(vecs[a] @ vecs[b])  # noqa: E731
    assert sim("t-0001", "t-0003") > sim("t-0001", "t-0002")
