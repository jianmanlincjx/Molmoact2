#!/usr/bin/env python3
"""Continuously compare per-task LIBERO success rates across checkpoints."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
import time
from typing import Any


SUITE_ORDER = {
    "libero_spatial": 0,
    "libero_object": 1,
    "libero_10": 2,
    "libero_goal": 3,
}


@dataclass(frozen=True)
class Checkpoint:
    label: str
    root: Path


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def task_key(suite: str, task_id: int) -> str:
    return f"{suite}:{task_id}"


def metric(successes: int, episodes: int) -> dict[str, int | float]:
    return {
        "successes": successes,
        "episodes": episodes,
        "pc_success": 100.0 * successes / episodes if episodes else 0.0,
    }


def collect_checkpoint(checkpoint: Checkpoint) -> dict[str, Any]:
    tasks: dict[str, dict[str, Any]] = {}
    suite_status: dict[str, str] = {}
    total_successes = 0
    total_episodes = 0

    for live_path in checkpoint.root.glob("*/live_eval.json"):
        suite = live_path.parent.name
        live = load_json(live_path) or {}
        suite_status[suite] = str(live.get("status", "pending"))
        for task in live.get("per_task", []):
            task_suite = str(task["task_group"])
            task_id = int(task["task_id"])
            successes = [
                bool(value)
                for value in task.get("metrics", {}).get("successes", [])
            ]
            success_count = sum(successes)
            episodes = len(successes)
            total_successes += success_count
            total_episodes += episodes
            tasks[task_key(task_suite, task_id)] = {
                "suite": task_suite,
                "task_id": task_id,
                "status": "final",
                "is_lower_bound": False,
                **metric(success_count, episodes),
            }

        realtime = load_json(live_path.parent / "realtime_accuracy.json") or {}
        for running in realtime.get("running_tasks", []):
            task_suite = str(running["task_group"])
            task_id = int(running["task_id"])
            key = task_key(task_suite, task_id)
            if key in tasks:
                continue
            successes = int(running.get("successes_so_far", 0))
            episodes = int(running.get("total_rollouts", 0))
            tasks[key] = {
                "suite": task_suite,
                "task_id": task_id,
                "status": "running",
                "is_lower_bound": True,
                "finished_rollouts": int(running.get("finished_rollouts", 0)),
                "step": int(running.get("step", 0)),
                "max_steps": int(running.get("max_steps", 0)),
                **metric(successes, episodes),
            }

    return {
        "label": checkpoint.label,
        "root": str(checkpoint.root),
        "tasks": tasks,
        "suite_status": suite_status,
        "completed": metric(total_successes, total_episodes),
    }


def row_sort_key(key: str) -> tuple[int, int, str]:
    suite, task_id = key.rsplit(":", 1)
    return (SUITE_ORDER.get(suite, 99), int(task_id), suite)


def build_payload(checkpoints: list[Checkpoint]) -> dict[str, Any]:
    collected = [collect_checkpoint(checkpoint) for checkpoint in checkpoints]
    keys = sorted(
        {key for result in collected for key in result["tasks"]},
        key=row_sort_key,
    )
    rows = []
    for key in keys:
        suite, task_id = key.rsplit(":", 1)
        values = {
            result["label"]: result["tasks"].get(key)
            for result in collected
        }
        rows.append({"suite": suite, "task_id": int(task_id), "checkpoints": values})
    return {
        "checkpoints": [
            {
                key: value
                for key, value in result.items()
                if key != "tasks"
            }
            for result in collected
        ],
        "rows": rows,
        "updated_at": time.time(),
    }


def format_rate(value: dict[str, Any] | None) -> str:
    if value is None:
        return "—"
    suffix = " lower bound" if value.get("is_lower_bound") else ""
    return (
        f"{value['pc_success']:.2f}% "
        f"({value['successes']}/{value['episodes']}){suffix}"
    )


def render_html(payload: dict[str, Any], refresh_seconds: int) -> str:
    labels = [checkpoint["label"] for checkpoint in payload["checkpoints"]]
    summaries = []
    for checkpoint in payload["checkpoints"]:
        complete = checkpoint["completed"]
        summaries.append(
            "<div class='summary'>"
            f"<h2>{escape(checkpoint['label'])}</h2>"
            f"<strong>{complete['pc_success']:.2f}%</strong>"
            f"<span>{complete['successes']}/{complete['episodes']} completed rollouts</span>"
            "</div>"
        )

    header = "".join(
        f"<th>{escape(label)} success</th>" for label in labels
    )
    body_rows = []
    for row in payload["rows"]:
        cells = []
        for label in labels:
            value = row["checkpoints"].get(label)
            status_class = "running" if value and value.get("status") == "running" else ""
            cells.append(f"<td class='{status_class}'>{escape(format_rate(value))}</td>")

        body_rows.append(
            "<tr>"
            f"<td>{escape(row['suite'])}</td>"
            f"<td>{row['task_id']}</td>"
            + "".join(cells)
            + "</tr>"
        )

    updated = datetime.fromtimestamp(payload["updated_at"]).strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{refresh_seconds}">
  <title>LIBERO checkpoint task comparison</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 28px; color: #222; }}
    h1 {{ margin-bottom: 4px; }}
    .caption {{ color: #666; margin-bottom: 20px; }}
    .summaries {{ display: flex; gap: 24px; margin: 20px 0; }}
    .summary {{ border: 1px solid #ddd; padding: 12px 18px; min-width: 220px; }}
    .summary h2 {{ font-size: 16px; margin: 0 0 8px; }}
    .summary strong {{ display: block; font-size: 24px; }}
    .summary span {{ color: #666; font-size: 13px; }}
    table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    thead {{ position: sticky; top: 0; background: #f5f5f5; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    .running {{ color: #8a5a00; }}
  </style>
</head>
<body>
  <h1>LIBERO per-task checkpoint comparison</h1>
  <div class="caption">Updated {updated} · auto-refresh every {refresh_seconds}s · running rates are lower bounds</div>
  <div class="summaries">{''.join(summaries)}</div>
  <table>
    <thead><tr><th>Suite</th><th>Task ID</th>{header}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</body>
</html>
"""


def render_markdown(payload: dict[str, Any]) -> str:
    labels = [checkpoint["label"] for checkpoint in payload["checkpoints"]]
    updated = datetime.fromtimestamp(payload["updated_at"]).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# LIBERO per-task checkpoint comparison",
        "",
        f"Updated: `{updated}` · running success rates are lower bounds",
        "",
        "## Overall completed rollouts",
        "",
    ]
    for checkpoint in payload["checkpoints"]:
        complete = checkpoint["completed"]
        lines.append(
            f"- **{checkpoint['label']}**: {complete['pc_success']:.2f}% "
            f"({complete['successes']}/{complete['episodes']})"
        )

    header = ["Suite", "Task ID", *[f"{label} success" for label in labels]]
    lines.extend(
        [
            "",
            "## Per-task success rate",
            "",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
    )
    for row in payload["rows"]:
        cells = [row["suite"], str(row["task_id"])]
        for label in labels:
            value = row["checkpoints"].get(label)
            if value is None:
                cells.append("—")
            elif value.get("is_lower_bound"):
                cells.append(
                    f"≥{value['pc_success']:.2f}% "
                    f"({value['successes']}/{value['episodes']}, running)"
                )
            else:
                cells.append(
                    f"{value['pc_success']:.2f}% "
                    f"({value['successes']}/{value['episodes']})"
                )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_csv(payload: dict[str, Any], path: Path) -> None:
    labels = [checkpoint["label"] for checkpoint in payload["checkpoints"]]
    fields = ["suite", "task_id"]
    for label in labels:
        fields.extend(
            [
                f"{label}_status",
                f"{label}_successes",
                f"{label}_episodes",
                f"{label}_pc_success",
                f"{label}_is_lower_bound",
            ]
        )
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in payload["rows"]:
            output: dict[str, Any] = {
                "suite": row["suite"],
                "task_id": row["task_id"],
            }
            for label in labels:
                value = row["checkpoints"].get(label)
                if value:
                    for field in ("status", "successes", "episodes", "pc_success", "is_lower_bound"):
                        output[f"{label}_{field}"] = value.get(field)
            writer.writerow(output)
    os.replace(temporary, path)


def parse_checkpoint(value: str) -> Checkpoint:
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=/path/to/eval/root") from error
    if not label or not path:
        raise argparse.ArgumentTypeError("checkpoint must contain both label and path")
    return Checkpoint(label=label, root=Path(path))


def update(args: argparse.Namespace) -> None:
    payload = build_payload(args.checkpoint)
    if args.output.suffix.lower() in {".md", ".markdown"}:
        # Keep the same inode so Cursor/VS Code Markdown previews do not lose the
        # open document when the live table refreshes.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_markdown(payload), encoding="utf-8")
    else:
        atomic_text(args.output, render_html(payload, int(args.interval)))
    atomic_text(args.json, json.dumps(payload, indent=2) + "\n")
    write_csv(payload, args.csv)
    row_count = len(payload["rows"])
    summary = " | ".join(
        f"{checkpoint['label']}={checkpoint['completed']['pc_success']:.2f}%"
        for checkpoint in payload["checkpoints"]
    )
    print(f"[checkpoint-compare] tasks={row_count} {summary} output={args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=parse_checkpoint, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if len(args.checkpoint) < 2:
        raise ValueError("at least two --checkpoint values are required")
    if args.csv is None:
        args.csv = args.output.with_suffix(".csv")
    if args.json is None:
        args.json = args.output.with_suffix(".json")
    if args.interval <= 0:
        raise ValueError("--interval must be positive")

    while True:
        update(args)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
