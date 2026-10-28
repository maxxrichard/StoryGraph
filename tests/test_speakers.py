from storygraph.schemas import Kind, Utterance
from storygraph.understanding.speakers import infer_roles, prune, speaker_shares


def test_speaker_shares_and_roles():
    utts = [
        Utterance(text="one two three four five six", start=0, end=1, speaker="A"),
        Utterance(text="seven eight", start=1, end=2, speaker="B"),
        Utterance(text="nine", start=2, end=3),
    ]
    shares = speaker_shares(utts)
    assert abs(sum(shares.values()) - 1) < 1e-9 and shares["A"] > shares["B"]
    roles = {p.speaker_id: p.role for p in infer_roles(shares, Kind.AUDIOBOOK)}
    assert roles["A"] == "narrator" and roles["B"] == "character"
    roles = {p.speaker_id: p.role for p in infer_roles(shares, Kind.PODCAST)}
    assert roles["A"] == "host" and roles["B"] == "guest"
    assert [p.speaker_id for p in prune(infer_roles(shares, Kind.PODCAST), min_share=0.2)] == [
        "A",
        "B",
    ]
