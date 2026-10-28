import pytest
from pydantic import ValidationError

from storygraph.schemas import ContentSignals, Label, Segment, Title, Transcript


def test_transcript_from_text_has_monotone_timings():
    tr = Transcript.from_text("t", "First paragraph here.\nSecond one here.\n\nThird.")
    assert len(tr.utterances) == 3
    assert all(a.end <= b.start for a, b in zip(tr.utterances, tr.utterances[1:]))
    assert tr.duration == tr.utterances[-1].end
    assert "Second" in tr.text


def test_content_key_is_stable_and_sensitive():
    a = Title(title_id="x", title="A", duration_min=1, transcript_path="p")
    b = Title(title_id="x", title="A", duration_min=1, transcript_path="p")
    c = Title(title_id="x", title="A", duration_min=1, transcript_path="q")
    assert a.content_key == b.content_key != c.content_key


def test_signals_dedupe_and_lowercase_labels():
    s = ContentSignals(
        title_id="t",
        genres=[
            Label(name="Crime", confidence=0.4),
            Label(name="crime", confidence=0.9),
            Label(name="horror", confidence=0.5),
        ],
    )
    assert s.label_names("genres") == ["crime", "horror"]
    assert s.genres[0].confidence == 0.9


def test_signals_enforce_limits():
    with pytest.raises(ValidationError):
        ContentSignals(title_id="t", genres=[Label(name=f"g{i}", confidence=0.5) for i in range(4)])
    with pytest.raises(ValidationError):
        ContentSignals(title_id="t", tension=1.5)


def test_segment_rejects_inverted_bounds():
    with pytest.raises(ValidationError):
        Segment(title_id="t", index=0, start=5, end=1, text="x", word_count=1)


def test_signals_json_schema_is_guided_decoding_friendly():
    schema = ContentSignals.model_json_schema()
    assert "properties" in schema and "genres" in schema["properties"]
