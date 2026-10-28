"""Spark feature-store jobs (run on EMR Serverless from the Step Functions pipeline).

Imports pyspark lazily so the module can be imported and unit-tested without a Spark install;
the per-row logic is factored into pure functions reused by the serving path.
"""

from __future__ import annotations

import argparse
from typing import Any


def build_title_signals(
    spark: Any, signals_uri: str, out_uri: str, shard_count: int = 256
) -> None:  # pragma: no cover - needs Spark
    from pyspark.sql import functions as F

    df = spark.read.json(signals_uri)
    df = (
        df.withColumn("shard", F.abs(F.hash("title_id")) % shard_count)
        .withColumn("genre_primary", F.element_at("genres", 1))
        .withColumn("speaker_count", F.coalesce(F.col("speaker_count"), F.lit(1)))
    )
    df.write.mode("overwrite").partitionBy("shard").format("delta").save(out_uri)


def build_listener_features(
    spark: Any, interactions_uri: str, titles_uri: str, out_uri: str, as_of: str
) -> None:  # pragma: no cover - needs Spark
    from pyspark.sql import functions as F

    inter = spark.read.parquet(interactions_uri).filter(F.col("ts") < F.lit(as_of))
    titles = (
        spark.read.format("delta")
        .load(titles_uri)
        .select("title_id", "genres", "moods", "narrator")
    )
    joined = inter.join(titles, "title_id", "left")
    age_days = F.datediff(F.lit(as_of), F.to_date("ts"))
    w = F.pow(F.lit(0.5), age_days / F.lit(30.0))
    engagement = F.least(
        F.lit(1.0), F.col("listened_seconds") / F.greatest(F.lit(1.0), F.col("started_seconds"))
    )
    exploded = joined.select(
        "listener_id", F.explode_outer("genres").alias("genre"), (w * engagement).alias("wt")
    )
    genre_aff = exploded.groupBy("listener_id").pivot("genre").agg(F.sum("wt"))
    stats = joined.groupBy("listener_id").agg(
        (F.sum(F.when(age_days <= 30, F.col("listened_seconds")).otherwise(0.0)) / 3600).alias(
            "listen_hours_30d"
        ),
        F.avg(F.when(age_days <= 90, F.col("finished").cast("double"))).alias("finish_rate_90d"),
        F.collect_set("narrator").alias("recent_narrators"),
        F.collect_list("title_id").alias("history"),
    )
    stats.join(genre_aff, "listener_id", "left").write.mode("overwrite").format("delta").save(
        out_uri
    )


def build_ranker_dataset(
    spark: Any, exposures_uri: str, interactions_uri: str, out_uri: str, cutoff: str
) -> None:  # pragma: no cover - needs Spark
    """Join impressions with outcomes to produce (group, features, graded label) rows for LightGBM."""
    from pyspark.sql import functions as F

    exp = spark.read.parquet(exposures_uri).filter(F.col("ts") < F.lit(cutoff))
    out = spark.read.parquet(interactions_uri)
    joined = exp.join(out, ["listener_id", "title_id"], "left").fillna(
        {"listened_seconds": 0.0, "started_seconds": 0.0, "finished": False, "converted": False}
    )
    ratio = F.col("listened_seconds") / F.greatest(F.lit(1.0), F.col("started_seconds"))
    label = (
        F.when(F.col("finished") | F.col("converted"), 3)
        .when(ratio >= 0.6, 2)
        .when(ratio >= 0.1, 1)
        .otherwise(0)
    )
    joined.withColumn("label", label).write.mode("overwrite").parquet(out_uri)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI for EMR
    p = argparse.ArgumentParser()
    p.add_argument(
        "job", choices=["build-title-signals", "build-listener-features", "build-ranker-dataset"]
    )
    p.add_argument("--run-id", required=True)
    p.add_argument("--root", default="s3://storygraph-artifacts")
    p.add_argument("--as-of", default=None)
    args = p.parse_args(argv)
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName(f"storygraph-{args.job}").getOrCreate()
    if args.job == "build-title-signals":
        build_title_signals(
            spark,
            f"{args.root}/content/runs/{args.run_id}/signals/*.jsonl",
            "s3://storygraph-feature-store/title_signals",
        )
    elif args.job == "build-listener-features":
        build_listener_features(
            spark,
            f"{args.root}/interactions",
            "s3://storygraph-feature-store/title_signals",
            "s3://storygraph-feature-store/listener_features",
            args.as_of or "",
        )
    else:
        build_ranker_dataset(
            spark,
            f"{args.root}/exposures",
            f"{args.root}/interactions",
            f"{args.root}/ranker-dataset/{args.run_id}",
            args.as_of or "",
        )


if __name__ == "__main__":  # pragma: no cover
    main()
