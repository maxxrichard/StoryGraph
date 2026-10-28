import numpy as np

from storygraph.schemas import ContentSignals, NarrativeArc, Segment
from storygraph.understanding.narrative import classify_arc, lexicon_tension, tension_curve


def _seg(i, text, words=100):
    return Segment(title_id="t", index=i, start=0, end=1, text=text, word_count=words)


def test_lexicon_tension_orders_texts():
    scary = "fear terror scream blood dead panic footsteps dark"
    calm = "laughing warm gold songs kindness home quiet morning"
    assert lexicon_tension(scary) > 0.5 > lexicon_tension(calm)
    assert lexicon_tension("") == 0.0


def test_tension_curve_blends_llm_and_lexicon():
    segs = [_seg(0, "calm warm", 100), _seg(1, "terror panic", 100)]
    sigs = [ContentSignals(title_id="t", tension=0.1), ContentSignals(title_id="t", tension=0.9)]
    curve = tension_curve(segs, sigs, bins=4)
    assert curve.shape == (4,)
    assert curve[0] < curve[-1]
    assert np.allclose(tension_curve([], None, bins=4), 0)


def test_classify_arc_templates():
    assert classify_arc(np.linspace(0, 1, 12)) == NarrativeArc.RICHES_TO_RAGS
    assert classify_arc(np.linspace(1, 0, 12)) == NarrativeArc.RAGS_TO_RICHES
    hole = np.concatenate([np.linspace(0, 1, 6), np.linspace(1, 0, 6)])
    assert classify_arc(hole) == NarrativeArc.MAN_IN_HOLE
    assert classify_arc(np.full(12, 0.5)) == NarrativeArc.EXPOSITORY
    assert classify_arc(np.zeros(0)) == NarrativeArc.EXPOSITORY
    noise = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], float)
    assert classify_arc(noise) == NarrativeArc.EPISODIC
