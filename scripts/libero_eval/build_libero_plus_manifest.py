#!/usr/bin/env python3
"""Build a deterministic, language-free LIBERO-Plus balanced manifest."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


SUITES = ("libero_object", "libero_10", "libero_goal", "libero_spatial")
CATEGORIES = (
    "Background Textures",
    "Camera Viewpoints",
    "Light Conditions",
    "Objects Layout",
    "Robot Initial States",
    "Sensor Noise",
)
LEVELS = (1, 2, 3, 4, 5)
LANGUAGE_CATEGORY = "Language Instructions"


def load_classification(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = set(SUITES) - payload.keys()
    if missing:
        raise ValueError(f"Classification is missing suites: {sorted(missing)}")
    for suite in SUITES:
        ids = [entry["id"] for entry in payload[suite]]
        if ids != list(range(1, len(ids) + 1)):
            raise ValueError(f"{suite}: classification IDs must be contiguous and one-based")
    return payload


def select_balanced(
    data: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    samples_per_cell: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for suite_index, suite in enumerate(SUITES):
        for category_index, category in enumerate(CATEGORIES):
            selected_ids: set[int] = set()
            deficits: list[tuple[int, int]] = []
            for level in LEVELS:
                candidates = [
                    entry
                    for entry in data[suite]
                    if entry["category"] == category and entry.get("difficulty_level") == level
                ]
                if not candidates:
                    raise ValueError(f"No tasks for {suite}/{category}/level-{level}")
                count = min(samples_per_cell, len(candidates))
                cell_seed = seed + suite_index * 10_000 + category_index * 100 + level
                sampled = random.Random(cell_seed).sample(candidates, count)
                for entry in sampled:
                    selected.append(_record(suite, entry, level))
                    selected_ids.add(entry["id"])
                if count < samples_per_cell:
                    deficits.append((level, samples_per_cell - count))

            for target_level, count in deficits:
                available = [
                    entry
                    for entry in data[suite]
                    if entry["category"] == category and entry["id"] not in selected_ids
                ]
                rng = random.Random(
                    seed + suite_index * 10_000 + category_index * 100 + target_level + 50
                )
                rng.shuffle(available)
                available.sort(
                    key=lambda entry: abs((entry.get("difficulty_level") or 99) - target_level)
                )
                backfill = available[:count]
                if len(backfill) != count:
                    raise ValueError(f"Cannot backfill {suite}/{category}/level-{target_level}")
                for entry in backfill:
                    selected.append(_record(suite, entry, target_level))
                    selected_ids.add(entry["id"])
    return sorted(selected, key=lambda item: (item["suite"], item["task_id"]))


def _record(suite: str, entry: dict[str, Any], sampling_level: int) -> dict[str, Any]:
    return {
        "suite": suite,
        "classification_id": entry["id"],
        "task_id": entry["id"] - 1,
        "name": entry["name"],
        "category": entry["category"],
        "difficulty_level": entry.get("difficulty_level"),
        "sampling_level": sampling_level,
    }


def write_manifest(
    output: Path,
    *,
    classification: Path,
    seed: int,
    samples_per_cell: int,
    episodes_per_task: int,
) -> None:
    tasks = select_balanced(
        load_classification(classification),
        seed=seed,
        samples_per_cell=samples_per_cell,
    )
    expected = len(SUITES) * len(CATEGORIES) * len(LEVELS) * samples_per_cell
    if len(tasks) != expected:
        raise ValueError(f"Expected {expected} tasks, selected {len(tasks)}")
    if any(task["category"] == LANGUAGE_CATEGORY or "_language_" in task["name"] for task in tasks):
        raise ValueError("Language-perturbed task leaked into the balanced manifest")

    manifest = {
        "protocol_version": 1,
        "phase": "balanced",
        "selection_seed": seed,
        "rollout_seed": seed,
        "samples_per_cell": samples_per_cell,
        "episodes_per_task": episodes_per_task,
        "excluded_categories": [LANGUAGE_CATEGORY],
        "included_categories": list(CATEGORIES),
        "suites": list(SUITES),
        "num_tasks": len(tasks),
        "num_rollouts": len(tasks) * episodes_per_task,
        "tasks": tasks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tasks[0]))
        writer.writeheader()
        writer.writerows(tasks)

    print(
        f"[manifest] tasks={len(tasks)} rollouts={manifest['num_rollouts']} "
        f"suites={dict(Counter(task['suite'] for task in tasks))}"
    )
    print(f"[manifest] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--samples-per-cell", type=int, default=3)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    args = parser.parse_args()
    write_manifest(
        args.output,
        classification=args.classification,
        seed=args.seed,
        samples_per_cell=args.samples_per_cell,
        episodes_per_task=args.episodes_per_task,
    )


if __name__ == "__main__":
    main()
