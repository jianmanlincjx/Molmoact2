#!/usr/bin/env python3
"""Verify every data parquet in a derived dataset is actually readable.

goal_pose_provenance.json records SHA-256 for only 7 metadata artifacts; the
data/**.parquet files are NOT covered. A truncated transfer therefore passes the
driver's provenance check and dies ~70 s into the run, inside the dataloader,
after the job has already claimed the GPUs. Reading each footer here costs a few
seconds and turns that into an immediate refusal.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--expect-rows", type=int, default=None)
    args = ap.parse_args()

    data_dir = os.path.join(args.dataset_root, "data")
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    if not files:
        print(f"REFUSING: no parquet files under {data_dir}", file=sys.stderr)
        return 2

    corrupt: list[tuple[str, str]] = []
    rows = 0
    for path in files:
        try:
            rows += pq.ParquetFile(path).metadata.num_rows
        except Exception as exc:  # noqa: BLE001 - any failure to read is fatal here
            corrupt.append((os.path.relpath(path, args.dataset_root), type(exc).__name__))

    if corrupt:
        print(
            f"REFUSING: {len(corrupt)} of {len(files)} data parquet files are unreadable "
            "(truncated or corrupt transfer):",
            file=sys.stderr,
        )
        for rel, err in corrupt:
            print(f"  {rel}  [{err}]", file=sys.stderr)
        return 2

    if args.expect_rows is not None and rows != args.expect_rows:
        print(
            f"REFUSING: data has {rows} rows, expected {args.expect_rows}",
            file=sys.stderr,
        )
        return 2

    print(f"[integrity] {len(files)} parquet files readable, {rows:,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
