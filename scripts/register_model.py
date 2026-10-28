#!/usr/bin/env python
"""Register a trained recommender version (towers + ranker) under a model alias in S3 / the registry."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", required=True)
    p.add_argument("--model-dir", default="/opt/ml/processing/model")
    p.add_argument("--registry", default="s3://storygraph-models/recsys")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    manifest = {
        "version": args.version,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "files": sorted(str(f.name) for f in Path(args.model_dir).glob("*"))
        if Path(args.model_dir).exists()
        else [],
    }
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        return 0
    try:
        import boto3
    except ImportError:
        print("boto3 not installed; nothing uploaded", file=sys.stderr)
        return 0
    bucket, _, prefix = args.registry.removeprefix("s3://").partition("/")
    s3 = boto3.client("s3")
    for f in Path(args.model_dir).glob("*"):
        s3.upload_file(str(f), bucket, f"{prefix}/{args.version}/{f.name}")
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/{args.version}/manifest.json",
        Body=json.dumps(manifest).encode(),
    )
    s3.put_object(Bucket=bucket, Key=f"{prefix}/latest", Body=args.version.encode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
