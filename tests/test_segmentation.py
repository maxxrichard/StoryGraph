import numpy as np

from storygraph.schemas import Transcript
from storygraph.understanding.segmentation import (
    depth_scores,
    pick_boundaries,
    segment_transcript,
    split_sentences,
    texttiling_scores,
)


def _two_topic_text(n: int = 15) -> str:
    sea = " ".join(
        ["The lighthouse keeper watched the fog roll over the grey sea and the empty boat."] * n
    )
    bank = " ".join(
        ["The regulator asked the bank about deposits, capital and the treasury desk memo."] * n
    )
    return sea + " " + bank


def test_split_sentences():
    assert split_sentences("One. Two! Three? Four.") == ["One.", "Two!", "Three?", "Four."]


def test_texttiling_finds_topic_shift():
    sents = split_sentences(_two_topic_text())
    sim = texttiling_scores(sents, window=4)
    depth = depth_scores(sim, smoothing=1)
    bounds = pick_boundaries(depth)
    assert any(abs(b - 15) <= 1 for b in bounds), bounds


def test_segment_transcript_respects_bounds(sample_transcripts, small_seg_cfg):
    tr = sample_transcripts["t-0001"]
    segs = segment_transcript(
        tr, window_sentences=4, min_segment_words=80, max_segment_words=400, smoothing=1
    )
    assert len(segs) >= 2
    assert all(
        s.word_count <= 400 + 60 for s in segs
    )  # a single long sentence may overshoot slightly
    assert segs[0].start == 0.0
    assert all(a.end <= b.start + 1e-6 for a, b in zip(segs, segs[1:]))
    assert abs(segs[-1].end - tr.duration) < 1e-6
    assert " ".join(s.text for s in segs).split() == tr.text.split()


def test_embedding_breakpoints_method(embedder):
    tr = Transcript.from_text("t", _two_topic_text())
    segs = segment_transcript(
        tr,
        method="embedding-breakpoints",
        window_sentences=4,
        min_segment_words=50,
        max_segment_words=5000,
        smoothing=1,
        embed=embedder.encode,
    )
    assert len(segs) == 2
    assert "lighthouse" in segs[0].text and "regulator" in segs[1].text


def test_empty_transcript_yields_no_segments():
    assert segment_transcript(Transcript(title_id="t", utterances=[])) == []


def test_speaker_shares_propagate():
    tr = Transcript(
        title_id="t",
        utterances=[
            {
                "text": "Hello there everyone welcome to the show.",
                "start": 0,
                "end": 5,
                "speaker": "A",
            },
            {"text": "Thanks for having me on today.", "start": 5, "end": 9, "speaker": "B"},
        ],
    )
    segs = segment_transcript(tr, min_segment_words=1, max_segment_words=100)
    assert set(segs[0].speakers) == {"A", "B"}
    assert abs(sum(segs[0].speakers.values()) - 1) < 1e-6
    assert np.isclose(segs[0].speakers["A"], 7 / 13)
