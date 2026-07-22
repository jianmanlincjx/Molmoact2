#!/usr/bin/env python3
"""Monitor running LIBERO evaluations and maintain an aggregate live summary."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _metric(successes: list[bool]) -> dict[str, int | float]:
    success_count = sum(bool(value) for value in successes)
    count = len(successes)
    return {
        "successes": success_count,
        "episodes": count,
        "pc_success": 100.0 * success_count / count if count else 0.0,
    }


def collect(eval_root: Path) -> dict[str, Any]:
    manifest = _load_json(eval_root / "task_manifest.json")
    metadata = {}
    if manifest:
        metadata = {
            (task["suite"], int(task["task_id"])): task for task in manifest.get("tasks", [])
        }

    suites: dict[str, Any] = {}
    all_successes: list[bool] = []
    by_category: dict[str, list[bool]] = defaultdict(list)
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
    completed_tasks = 0
    total_tasks = 0
    statuses: list[str] = []

    suite_names = manifest.get("suites", []) if manifest else []
    suite_names = sorted(
        set(suite_names)
        | {path.parent.name for path in eval_root.glob("*/live_eval.json")}
        | {path.parent.name for path in eval_root.glob("*/process_status.json")}
    )
    for suite in suite_names:
        suite_dir = eval_root / suite
        live = _load_json(suite_dir / "live_eval.json") or {}
        process = _load_json(suite_dir / "process_status.json") or {"status": "pending"}
        status = "final" if live.get("status") == "final" else process.get("status", "pending")
        statuses.append(status)

        suite_successes: list[bool] = []
        for task in live.get("per_task", []):
            successes = [bool(value) for value in task.get("metrics", {}).get("successes", [])]
            suite_successes.extend(successes)
            key = (task["task_group"], int(task["task_id"]))
            task_meta = metadata.get(key)
            if task_meta:
                by_category[task_meta["category"]].extend(successes)
                by_difficulty[str(task_meta["sampling_level"])].extend(successes)

        all_successes.extend(suite_successes)
        suite_completed = int(live.get("completed_tasks", 0))
        suite_total = int(live.get("total_tasks", 0))
        if not suite_total and manifest:
            suite_total = sum(task["suite"] == suite for task in manifest["tasks"])
        completed_tasks += suite_completed
        total_tasks += suite_total
        suites[suite] = {
            "status": status,
            "completed_tasks": suite_completed,
            "total_tasks": suite_total,
            **_metric(suite_successes),
        }

    overall_status = "running"
    if statuses and all(status in {"final", "failed"} for status in statuses):
        overall_status = "failed" if "failed" in statuses else "final"
    return {
        "status": overall_status,
        "completed_tasks": completed_tasks,
        "total_tasks": total_tasks,
        "overall": _metric(all_successes),
        "per_suite": suites,
        "per_category": {key: _metric(values) for key, values in sorted(by_category.items())},
        "per_difficulty": {key: _metric(values) for key, values in sorted(by_difficulty.items())},
        "updated_at": time.time(),
    }


def _format_row(name: str, payload: dict[str, Any]) -> str:
    return (
        f"{name:<24} {payload.get('status', ''):<9} "
        f"tasks {payload.get('completed_tasks', '-'):>3}/{payload.get('total_tasks', '-'):<3} "
        f"rollouts {payload['episodes']:>5}  "
        f"success {payload['successes']:>4}  acc {payload['pc_success']:6.2f}%"
    )


def render(summary: dict[str, Any], eval_root: Path) -> str:
    overall = summary["overall"]
    lines = [
        f"Evaluation: {eval_root}",
        f"Status: {summary['status']}  tasks {summary['completed_tasks']}/{summary['total_tasks']}  "
        f"rollouts {overall['episodes']}  success {overall['successes']}  "
        f"current accuracy {overall['pc_success']:.2f}%",
        "",
        "Per suite:",
    ]
    lines.extend(_format_row(name, payload) for name, payload in summary["per_suite"].items())
    if summary["per_category"]:
        lines.extend(["", "Per perturbation category:"])
        for name, payload in summary["per_category"].items():
            lines.append(
                f"{name:<24} rollouts {payload['episodes']:>5}  "
                f"success {payload['successes']:>4}  acc {payload['pc_success']:6.2f}%"
            )
    if summary["per_difficulty"]:
        lines.extend(["", "Per difficulty:"])
        for name, payload in summary["per_difficulty"].items():
            lines.append(
                f"level {name:<3} rollouts {payload['episodes']:>5}  "
                f"success {payload['successes']:>4}  acc {payload['pc_success']:6.2f}%"
            )
    lines.append("\nAccuracy is computed over completed rollouts only.")
    return "\n".join(lines)


def write_summary(eval_root: Path, summary: dict[str, Any]) -> None:
    output = eval_root / "live_summary.json"
    tmp = output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output)
    with (eval_root / "live_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("dimension", "name", "successes", "episodes", "pc_success"),
        )
        writer.writeheader()
        writer.writerow({"dimension": "overall", "name": "overall", **summary["overall"]})
        for dimension, key in (
            ("suite", "per_suite"),
            ("category", "per_category"),
            ("difficulty", "per_difficulty"),
        ):
            for name, payload in summary[key].items():
                writer.writerow(
                    {
                        "dimension": dimension,
                        "name": name,
                        **{field: payload[field] for field in ("successes", "episodes", "pc_success")},
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        summary = collect(args.eval_root)
        args.eval_root.mkdir(parents=True, exist_ok=True)
        write_summary(args.eval_root, summary)
        if not args.once and os.isatty(1):
            print("\033[2J\033[H", end="")
        print(render(summary, args.eval_root), flush=True)
        if args.once or summary["status"] in {"final", "failed"}:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
