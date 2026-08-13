#!/usr/bin/env python3
"""Synchronize a local training-data directory into an HF Storage Bucket.

Examples:
  python scripts/push_training_data.py training-data/96yottea
  python scripts/push_training_data.py training-data/96yottea \
      --remote-path datasets/96yottea --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_BUCKET = "brianw/training-data"


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE lines without replacing an existing environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def assert_bucket_access(bucket: str) -> None:
    """Fail early with a useful error instead of after building a sync plan."""
    try:
        HfApi(token=os.environ["HF_TOKEN"]).bucket_info(bucket)
    except Exception as error:
        raise RuntimeError(
            f"Cannot access HF Storage Bucket {bucket!r}. Check the namespace, "
            "that the bucket exists, and that HF_TOKEN has write access."
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local folder to upload")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="HF bucket ID")
    parser.add_argument(
        "--remote-path",
        help="Bucket prefix (default: datasets/<source directory name>)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show changes only")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete remote files absent locally (use only after reviewing --dry-run)",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        parser.error(f"source is not a directory: {source}")
    if not any(source.rglob("*")):
        parser.error(f"source is empty: {source}")

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not os.environ.get("HF_TOKEN"):
        parser.error("HF_TOKEN is required; put it in .env or export it")
    if shutil.which("hf") is None:
        parser.error("`hf` was not found; install requirements.txt first")
    try:
        assert_bucket_access(args.bucket)
    except RuntimeError as error:
        parser.error(str(error))

    remote_path = (args.remote_path or f"datasets/{source.name}").strip("/")
    destination = f"hf://buckets/{args.bucket}/{remote_path}"
    command = ["hf", "buckets", "sync", str(source), destination]
    if args.dry_run:
        command.append("--dry-run")
    if args.delete:
        command.append("--delete")

    print(f"Syncing {source} -> {destination}")
    subprocess.run(command, check=True, env=os.environ.copy())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
