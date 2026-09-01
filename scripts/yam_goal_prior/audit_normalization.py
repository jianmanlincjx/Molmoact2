#!/usr/bin/env python3
"""Audit the quantile normalization the YAM goal-pose trainer will actually apply.

This deliberately *consumes* the saved q01/q99 rather than recomputing them --
recomputation is ``prepare_dataset.py --verify-only``'s job. What is measured
here is the behaviour training sees: how much of each dimension gets clamped to
+/-1, and whether normalize -> clamp -> denormalize round-trips.

This is the gate for the defect that produced LIBERO v3: LeRobot v3.0 builds
``meta/stats.json`` by aggregating per-episode statistics, which is not the true
global quantile, and on LIBERO that clipped ~94% of the state-Z target to +/-1.
Correct quantile normalization clips ~2% per dimension by construction, so
anything materially above that means the statistics are wrong, not the data.

The defect is live in this data. Of the three merged sources only
``blocks_filtered`` escapes it (2.0% worst-dimension clipping); the shipped stats
for ``dustpan_filtered`` clip 24.2% of one state dimension and those for
``transfer_filtered`` 16.0%. ``merge_datasets.py`` and ``prepare_dataset.py`` both
recompute exact global quantiles, which brings the merged build back to 2.06% --
so this gate is what proves the recomputation actually happened.

Exit status is 1 when any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import (  # noqa: E402
    ACTION_DIM,
    ACTION_KEY,
    HORIZON,
    STATE_DIM,
    STATE_KEY,
    _episode_tables,
    _episode_arrays,
    _load_build_config,
    _valid_task_indices,
    anchor_mask,
    FilePlan,
)

FEATURE_DIMS = {STATE_KEY: STATE_DIM, ACTION_KEY: ACTION_DIM}
STAT_NAMES = ("count", "min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


class AuditError(RuntimeError):
    pass


_SCAN = [
    "index",
    "episode_index",
    "frame_index",
    "task_index",
    "timestamp",
    STATE_KEY,
    ACTION_KEY,
]


def _feature_names(info: dict[str, Any], key: str) -> list[str]:
    names = info.get("features", {}).get(key, {}).get("names")
    if isinstance(names, dict):
        flat: list[str] = []
        for value in names.values():
            flat.extend(value if isinstance(value, list) else [value])
        return [str(item) for item in flat]
    if isinstance(names, list):
        return [str(item) for item in names]
    raise AuditError(f"meta/info.json feature {key!r} has no per-dimension names")


def normalization_mask(info: dict[str, Any], key: str, dim: int) -> np.ndarray:
    """Rebuild the mask MolmoAct2 derives from names.

    ``processor_molmoact2._add_gripper_masks_to_stats`` normalizes a dimension
    only when "gripper" is absent from its name. Bimanual YAM has two grippers,
    unlike the single-gripper DROID/LIBERO layouts.
    """
    names = _feature_names(info, key)
    if len(names) != dim:
        raise AuditError(f"{key} declares {len(names)} names, expected {dim}")
    mask = np.asarray(["gripper" not in name.lower() for name in names], dtype=bool)
    grippers = int((~mask).sum())
    if grippers != 2:
        raise AuditError(f"{key} must name exactly two gripper dimensions, found {grippers}")
    return mask


def validate_feature_stats(stats: dict[str, Any], key: str, dim: int, expected_count: int) -> None:
    item = stats.get(key)
    if not isinstance(item, dict):
        raise AuditError(f"meta/stats.json is missing {key}")
    for name in STAT_NAMES:
        if name not in item:
            raise AuditError(f"{key} is missing statistic {name}")
    count = item["count"]
    if not isinstance(count, list) or len(count) != 1 or int(count[0]) != expected_count:
        raise AuditError(f"{key}.count must be [{expected_count}], got {count!r}")
    for name in STAT_NAMES[1:]:
        values = np.asarray(item[name], dtype=np.float64)
        if values.shape != (dim,):
            raise AuditError(f"{key}.{name} must have shape ({dim},), got {values.shape}")
        if not np.isfinite(values).all():
            raise AuditError(f"{key}.{name} contains a non-finite value")
    q01 = np.asarray(item["q01"], dtype=np.float64)
    q99 = np.asarray(item["q99"], dtype=np.float64)
    minimum = np.asarray(item["min"], dtype=np.float64)
    maximum = np.asarray(item["max"], dtype=np.float64)
    if not np.all(q99 > q01):
        raise AuditError(f"{key} has a non-positive q99-q01 span")
    if not np.all(q01 >= minimum) or not np.all(q99 <= maximum):
        raise AuditError(f"{key} quantiles fall outside min/max")


def audit(
    root: Path,
    outside_fraction_limit: float,
    roundtrip_atol: float,
) -> dict[str, Any]:
    provenance = json.loads((root / "goal_pose_provenance.json").read_text())
    config = _load_build_config(provenance["config"])
    excluded = frozenset(config.exclude_episodes)
    plans = [FilePlan(item["path"], int(item["output_rows"])) for item in provenance["files"]]
    info = json.loads((root / "meta/info.json").read_text())
    stats = json.loads((root / "meta/stats.json").read_text())
    anchor_count = int(provenance["anchor_count"])
    included_frames = int(provenance["included_frame_count"])

    for key, dim in FEATURE_DIMS.items():
        validate_feature_stats(stats, key, dim, included_frames)

    masks = {key: normalization_mask(info, key, dim) for key, dim in FEATURE_DIMS.items()}
    bounds = {
        key: (
            np.asarray(stats[key]["q01"], dtype=np.float64),
            np.asarray(stats[key]["q99"], dtype=np.float64),
        )
        for key in FEATURE_DIMS
    }

    accum = {
        key: {
            "count": 0,
            "below": np.zeros(dim, dtype=np.int64),
            "above": np.zeros(dim, dtype=np.int64),
            "min": np.full(dim, np.inf),
            "max": np.full(dim, -np.inf),
            "roundtrip": 0.0,
            "gripper_min": np.inf,
            "gripper_max": -np.inf,
        }
        for key, dim in FEATURE_DIMS.items()
    }

    def observe(key: str, values: np.ndarray, role: str = "") -> None:
        q01, q99 = bounds[key]
        mask = masks[key]
        state = accum[key]
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise AuditError(f"non-finite {key} in the training stream")
        state["count"] += len(values)
        state.setdefault("roles", set()).add(role)
        state["below"] += (values < q01).sum(axis=0)
        state["above"] += (values > q99).sum(axis=0)
        state["min"] = np.minimum(state["min"], values.min(axis=0))
        state["max"] = np.maximum(state["max"], values.max(axis=0))

        # Exactly what the processor does: quantile-normalize, leave gripper
        # dims raw, clamp to [-1, 1]; then invert and measure the error on the
        # dimensions that were not clamped away.
        normalized = 2.0 * (values - q01) / (q99 - q01) - 1.0
        normalized = np.where(mask, normalized, values)
        clamped = np.clip(normalized, -1.0, 1.0)
        restored = (clamped + 1.0) * (q99 - q01) / 2.0 + q01
        restored = np.where(mask, restored, clamped)
        inside = mask & (values >= q01) & (values <= q99)
        if inside.any():
            error = float(np.abs(restored[inside] - values[inside]).max())
            state["roundtrip"] = max(state["roundtrip"], error)
        gripper = values[:, ~mask]
        if gripper.size:
            state["gripper_min"] = min(state["gripper_min"], float(gripper.min()))
            state["gripper_max"] = max(state["gripper_max"], float(gripper.max()))

    valid_task_indices = _valid_task_indices(root)
    scanned = 0
    for episode_table in _episode_tables(root, plans, _SCAN):
        arrays = _episode_arrays(episode_table)
        mask, windows = anchor_mask(
            arrays,
            config.non_idle,
            valid_task_indices=valid_task_indices,
            excluded_episodes=excluded,
        )
        count = int(mask.sum())
        if not count:
            continue
        # The goal is observation.state at t+H, so the state statistic has to
        # cover both roles: the anchor frame and the frame the goal points at.
        observe(STATE_KEY, arrays[STATE_KEY][: len(mask)][mask], "current state")
        observe(STATE_KEY, arrays[STATE_KEY][HORIZON:][mask], f"goal state (t+{HORIZON})")
        observe(ACTION_KEY, windows[mask].reshape(-1, ACTION_DIM), "action chunk")
        scanned += count

    if scanned != anchor_count:
        raise AuditError(f"scanned {scanned} anchors, provenance records {anchor_count}")

    failures: list[str] = []
    report: dict[str, Any] = {"anchor_count": anchor_count, "features": {}}
    for key, dim in FEATURE_DIMS.items():
        state = accum[key]
        total = state["count"]
        outside = (state["below"] + state["above"]) / max(total, 1)
        names = _feature_names(info, key)
        mask = masks[key]
        recorded_min = np.asarray(stats[key]["min"], dtype=np.float64)
        recorded_max = np.asarray(stats[key]["max"], dtype=np.float64)
        # Statistics cover every frame; the training stream is a subset of those
        # frames, so it must stay inside the recorded envelope but need not
        # reach it exactly.
        if np.any(state["min"] < recorded_min - 1e-6):
            failures.append(f"{key}: training values fall below meta/stats.json min")
        if np.any(state["max"] > recorded_max + 1e-6):
            failures.append(f"{key}: training values exceed meta/stats.json max")
        worst = 0.0
        for index in range(dim):
            if not mask[index]:
                continue
            if outside[index] > outside_fraction_limit:
                failures.append(
                    f"{key}.{names[index]}: {100 * outside[index]:.2f}% of values clip, "
                    f"limit {100 * outside_fraction_limit:.1f}%"
                )
            worst = max(worst, float(outside[index]))
        if state["roundtrip"] > roundtrip_atol:
            failures.append(
                f"{key}: normalize/denormalize round-trip error {state['roundtrip']:.3e} "
                f"exceeds {roundtrip_atol:.0e}"
            )
        if state["gripper_min"] < -1.0 or state["gripper_max"] > 1.0:
            failures.append(
                f"{key}: un-normalized gripper leaves [-1, 1] "
                f"([{state['gripper_min']:.4f}, {state['gripper_max']:.4f}])"
            )
        report["features"][key] = {
            "samples": int(total),
            "worst_outside_fraction": worst,
            "outside_fraction": {names[i]: float(outside[i]) for i in range(dim)},
            "normalized_dims": int(mask.sum()),
            "raw_gripper_dims": [names[i] for i in range(dim) if not mask[i]],
            "gripper_range": [state["gripper_min"], state["gripper_max"]],
            "max_roundtrip_error": state["roundtrip"],
        }

    report["status"] = "failed" if failures else "passed"
    report["failures"] = failures
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--outside-fraction-limit",
        type=float,
        default=0.05,
        help="Maximum clip fraction for a non-gripper dimension (correct stats give ~0.02).",
    )
    parser.add_argument("--roundtrip-atol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = audit(
        args.dataset_root.expanduser().resolve(),
        args.outside_fraction_limit,
        args.roundtrip_atol,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")

    for key, item in report["features"].items():
        print(
            f"[audit] {key:<22} samples={item['samples']:>9}  "
            f"worst_clip={100 * item['worst_outside_fraction']:5.2f}%  "
            f"roundtrip={item['max_roundtrip_error']:.2e}  "
            f"raw_gripper={item['raw_gripper_dims']}"
        )
    for failure in report["failures"]:
        print(f"[audit] FAIL {failure}")
    print(f"[audit] {report['status'].upper()}")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
