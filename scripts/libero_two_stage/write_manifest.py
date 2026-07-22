#!/usr/bin/env python3
"""Write a reproducibility manifest for a LIBERO training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(command: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-type", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--command", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
    except ImportError:
        torch_info = None

    tracked_env = {
        key: os.environ.get(key)
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "HF_HOME",
            "HF_ENDPOINT",
            "TORCH_HOME",
            "UV_CACHE_DIR",
            "WANDB_PROJECT",
        )
    }
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_type": args.run_type,
        "seed": args.seed,
        "workspace": str(args.workspace.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "command": args.command,
        "git": {
            "sha": _run(["git", "rev-parse", "HEAD"], args.workspace),
            "branch": _run(["git", "branch", "--show-current"], args.workspace),
            "status": _run(["git", "status", "--short"], args.workspace),
            "diff_stat": _run(["git", "diff", "--stat"], args.workspace),
        },
        "dataset_hashes": {
            "info.json": _sha256(args.dataset_root / "meta" / "info.json"),
            "stats.json": _sha256(args.dataset_root / "meta" / "stats.json"),
            "tasks.jsonl": _sha256(args.dataset_root / "meta" / "tasks.jsonl"),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch_info,
            "nvidia_smi": _run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
                args.workspace,
            ),
        },
        "environment": tracked_env,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[manifest] wrote {args.output}")


if __name__ == "__main__":
    main()
