import numpy as np

from storygraph.ingest.asr import stitch, transcribe_chunks
from storygraph.ingest.audio import Chunk, chunk_audio, energy_vad
from storygraph.schemas import Utterance


def _signal(seconds: float, sr: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.2, int(seconds * sr)).astype(np.float32)
    # silence gaps every 10 s for 0.5 s
    for s in range(10, int(seconds), 10):
        y[s * sr : int((s + 0.5) * sr)] = 0.0
    return y


def test_energy_vad_marks_silence():
    sr = 1000
    y = _signal(30, sr)
    mask = energy_vad(y, sr, frame_ms=30)
    assert mask.size > 0 and mask.mean() > 0.8
    assert not mask[int(10.2 * sr / 30)]


def test_chunk_audio_covers_signal_with_overlap():
    sr = 1000
    y = _signal(65, sr)
    chunks = list(chunk_audio(y, sr, chunk_seconds=20, overlap_seconds=2, frame_ms=30))
    assert chunks[0].start == 0.0
    assert abs(chunks[-1].end - 65.0) < 1e-6
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end  # overlap
        assert b.start > a.start
    assert all(c.samples.size > 0 for c in chunks)


def test_stitch_drops_overlap_duplicates():
    utts = [
        Utterance(text="a", start=0, end=5),
        Utterance(text="b", start=5, end=10),
        Utterance(text="b-dup", start=6, end=10),  # from the next chunk's overlap region
        Utterance(text="c", start=10, end=15),
    ]
    out = stitch(utts, overlap_seconds=5)
    assert [u.text for u in out] == ["a", "b", "c"]


class _EchoBackend:
    name = "echo"

    def transcribe(self, chunk: Chunk, sr: int, language):
        return [Utterance(text=f"chunk{chunk.index}", start=chunk.start, end=chunk.end)]


def test_transcribe_chunks_builds_transcript():
    sr = 1000
    y = _signal(45, sr)
    chunks = chunk_audio(y, sr, chunk_seconds=20, overlap_seconds=2, snap_to_silence=False)
    tr = transcribe_chunks("t", chunks, sr, _EchoBackend(), overlap_seconds=2)
    assert tr.asr_model == "echo"
    assert len(tr.utterances) == 3
    assert tr.duration == 45.0
