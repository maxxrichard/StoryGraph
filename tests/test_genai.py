import json

import pytest

from storygraph.genai.judge import Calibration, SummaryJudge, calibrate, parse_judge, spearman
from storygraph.genai.summaries import (
    ChapterSummariser,
    DictCache,
    has_spoiler_markers,
    parse_summary,
)
from storygraph.schemas import Kind, Segment, Title
from storygraph.serving.llm_client import StubLLM


def _title():
    return Title(title_id="t", title="T", kind=Kind.AUDIOBOOK, duration_min=100)


def _seg(i=0):
    return Segment(
        title_id="t",
        index=i,
        start=0,
        end=600,
        text="The keeper watched the fog. " * 20,
        word_count=100,
    )


def test_spoiler_markers():
    assert has_spoiler_markers("It turns out the keeper was dead all along.")
    assert has_spoiler_markers("In the end she leaves.")
    assert not has_spoiler_markers("A keeper notices an unfamiliar boat.")


def test_parse_summary_defaults():
    s = parse_summary('{"heading": "", "summary": "  Two   spaces. "}', "t", 2)
    assert s.heading == "Chapter 3" and s.summary == "Two spaces." and s.spoiler_free
    with pytest.raises(ValueError):
        parse_summary("nope", "t", 0)


def test_summariser_caches_and_counts_calls():
    llm = StubLLM()
    cache = DictCache()
    s = ChapterSummariser(llm, cache=cache)
    a = s.summarise(_title(), _seg(), None, total=3)
    b = s.summarise(_title(), _seg(), None, total=3)
    assert a == b and llm.calls == 1
    assert a.heading and a.spoiler_free


class _SpoilerLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, system, user, max_tokens=256):
        self.calls += 1
        return json.dumps({"heading": "Reveal", "summary": "It turns out the brother is dead."})


def test_summariser_retries_then_blanks_spoilers():
    llm = _SpoilerLLM()
    out = ChapterSummariser(llm).summarise(_title(), _seg(), None, total=1)
    assert llm.calls == 2 and out.summary == "" and out.spoiler_free


def test_summarise_title_batches(sample_titles):
    s = ChapterSummariser(StubLLM())
    out = s.summarise_title(_title(), [_seg(0), _seg(1)])
    assert [o.segment_index for o in out] == [0, 1]


def test_judge_and_calibration():
    j = SummaryJudge(StubLLM())
    score = j.score("transcript text", "summary")
    assert score.overall == pytest.approx(4.25)
    parsed = parse_judge('{"faithfulness": 9, "spoiler_safety": 0, "coverage": 3, "fluency": 3}')
    assert parsed.faithfulness == 5 and parsed.spoiler_safety == 1
    with pytest.raises(ValueError):
        parse_judge("x")
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    cal = calibrate([4, 4.5, 3, 5, 2] * 12, [4, 4, 3, 5, 2.5] * 12)
    assert isinstance(cal, Calibration) and cal.trustworthy
    assert not calibrate([1, 2, 3], [3, 2, 1]).trustworthy
