#!/usr/bin/env python3
"""Aggregate aligned LIBERO evaluations and plot success-rate curves."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    for info_path in sorted(args.eval_root.glob("*/*/eval_info.json")):
        label = info_path.parent.parent.name
        suite = info_path.parent.name
        match = re.fullmatch(r"(baseline|two_stage)_total(\d+)k", label)
        if match is None:
            continue
        payload = json.loads(info_path.read_text(encoding="utf-8"))
        aggregate = payload.get("aggregated") or payload.get("overall")
        if not isinstance(aggregate, dict):
            raise ValueError(f"No aggregate metrics in {info_path}")
        rows.append(
            {
                "method": match.group(1),
                "total_steps": int(match.group(2)) * 1000,
                "suite": suite,
                "pc_success": float(aggregate["pc_success"]),
                "n_episodes": int(
                    aggregate.get("n_episodes")
                    or len(payload.get("per_episode", []))
                    or len(payload.get("episodes", []))
                ),
                "eval_info": str(info_path),
            }
        )
    if not rows:
        raise SystemExit(f"No aligned eval_info.json files found under {args.eval_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "success_rates.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "success_rates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    suites = sorted({row["suite"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, suite in zip(axes.flat, suites, strict=False):
        for method in ("baseline", "two_stage"):
            points = sorted(
                (row for row in rows if row["suite"] == suite and row["method"] == method),
                key=lambda row: row["total_steps"],
            )
            axis.plot(
                [row["total_steps"] for row in points],
                [row["pc_success"] for row in points],
                marker="o",
                label=method,
            )
        axis.set_title(suite)
        axis.set_xlabel("total optimizer steps")
        axis.set_ylabel("success rate (%)")
        axis.set_ylim(0, 100)
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "success_rate_curves.png", dpi=160)
    plt.close(fig)
    print(f"[summary] aggregated {len(rows)} evaluation results into {args.output_dir}")


if __name__ == "__main__":
    main()
