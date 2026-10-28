"""``storygraph`` command line: run the understanding pipeline locally, serve, analyse experiments."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="StoryGraph: content understanding + recommendations for long-form audio.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def understand(
    catalog: Path = typer.Argument(
        Path("data/sample/catalog.jsonl"), help="JSONL of Title records"
    ),
    output: Path = typer.Option(
        Path("artifacts/signals.jsonl"), help="Where to write title signal records"
    ),
    embedder: str = typer.Option("hashing", help="'hashing' or a sentence-transformers model name"),
    llm_endpoint: str | None = typer.Option(
        None, help="vLLM OpenAI-compatible base URL; stub LLM if omitted"
    ),
) -> None:
    """Run segmentation -> embeddings -> tagging -> narrative signals over a catalog."""
    from storygraph.config import SegmentationConfig
    from storygraph.schemas import Title, Transcript
    from storygraph.serving.llm_client import StubLLM, VLLMClient
    from storygraph.understanding.embeddings import get_embedder
    from storygraph.understanding.pipeline import understand_title
    from storygraph.understanding.tagger import SegmentTagger

    llm = VLLMClient(base_url=llm_endpoint, adapter="tagger") if llm_endpoint else StubLLM()
    tagger, emb = SegmentTagger(llm), get_embedder(embedder)
    seg_cfg = SegmentationConfig(
        window_sentences=4, min_segment_words=80, max_segment_words=400, smoothing=1
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table = Table(title="Title signals")
    for col in ("title_id", "segments", "genres", "themes", "moods", "arc"):
        table.add_column(col)
    n = 0
    with open(catalog, encoding="utf-8") as fh, open(output, "w", encoding="utf-8") as out:
        for line in fh:
            if not line.strip():
                continue
            t = Title(**json.loads(line))
            if not t.transcript_path or not Path(t.transcript_path).exists():
                console.print(f"[yellow]{t.title_id}: no transcript, skipped")
                continue
            tr = Transcript.from_text(
                t.title_id, Path(t.transcript_path).read_text(encoding="utf-8")
            )
            u = understand_title(t, tr, emb, tagger, seg_cfg)
            rec = u.to_record()
            out.write(json.dumps(rec) + "\n")
            table.add_row(
                t.title_id,
                str(rec["segment_count"]),
                ", ".join(rec["genres"]),
                ", ".join(rec["themes"][:4]),
                ", ".join(rec["moods"]),
                str(rec["narrative_arc"]),
            )
            n += 1
    console.print(table)
    console.print(f"[green]wrote {n} records to {output}")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8080, reload: bool = False) -> None:
    """Start the recommendation API (demo context from the sample catalog)."""
    import uvicorn

    uvicorn.run("storygraph.serving.api:app", host=host, port=port, reload=reload)


@app.command()
def analyze(
    exposures: Path = typer.Argument(
        ...,
        help="CSV with listener_id, variant, started_seconds, listened_seconds, finished, converted",
    ),
    metric: str = typer.Option("listen_through", help="listen_through | conversion | finish_rate"),
) -> None:
    """Analyse an A/B export: SRM check + Welch test on the listener-level metric."""
    import pandas as pd

    from storygraph.experiments.analysis import srm_check, welch_ttest
    from storygraph.experiments.metrics import per_listener, split

    units = per_listener(pd.read_csv(exposures).to_dict("records"))
    arms = split(units, metric)
    counts = {k: len(v) for k, v in arms.items()}
    ok, p = srm_check(counts, {k: 1 / len(counts) for k in counts})
    console.print(f"SRM check: {'ok' if ok else '[red]MISMATCH'} (p={p:.4f}, counts={counts})")
    if "control" in arms and "treatment" in arms:
        console.print(welch_ttest(arms["control"], arms["treatment"], metric=metric).summary())


@app.command()
def version() -> None:
    from storygraph import __version__

    console.print(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
