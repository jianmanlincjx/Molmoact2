from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.droid_goal_prior.audit_normalization import (
    ACTION_KEY,
    EE_POSE_KEY,
    STATE_KEY,
    AuditContractError,
    audit_feature_values,
    audit_normalization,
    normalization_mask_from_feature_meta,
    validate_feature_stats,
)


def _feature_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    return {
        "count": [len(values)],
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_full_recipe_fixture(root: Path) -> None:
    horizon = 15
    length = 115
    anchor_count = length - horizon
    data_path = root / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    (root / "meta").mkdir()

    rows: list[dict[str, object]] = []
    state_values = np.empty((length, 8), dtype=np.float32)
    action_values = np.empty((length, 8), dtype=np.float32)
    pose_values = np.empty((length, 7), dtype=np.float32)
    for frame in range(length):
        state = np.r_[
            np.arange(7, dtype=np.float32) * 0.1 + frame * 0.01,
            np.float32(frame / (length - 1)),
        ].astype(np.float32)
        action = np.r_[
            np.arange(7, dtype=np.float32) * 0.2 + frame * 0.015,
            np.float32(frame / (length - 1)),
        ].astype(np.float32)
        pose = np.r_[
            np.asarray([frame * 0.002, -frame * 0.001, 0.2 + frame * 0.001]),
            np.asarray([frame * 0.003, -frame * 0.002, frame * 0.001]),
            np.float32(frame / (length - 1)),
        ].astype(np.float32)
        state_values[frame] = state
        action_values[frame] = action
        pose_values[frame] = pose
        rows.append(
            {
                "index": frame,
                "episode_index": 0,
                "frame_index": frame,
                "task_index": 7,
                "timestamp": np.float32(frame / 15),
                "is_episode_successful": True,
                "language_instruction": "move the object",
                "language_instruction_2": "",
                "language_instruction_3": "",
                STATE_KEY: state.tolist(),
                EE_POSE_KEY: pose.tolist(),
                ACTION_KEY: action.tolist(),
                "action.joint_velocity": (
                    np.arange(7, dtype=np.float32) * 0.01 + frame * 0.01
                ).tolist(),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), data_path)
    pq.write_table(
        pa.table({"task_index": pa.array([7]), "task": ["move the object"]}),
        root / "meta/tasks.parquet",
    )

    manifest_path = root / "valid_anchor_indices.parquet"
    pq.write_table(
        pa.table(
            {
                "index": pa.array(np.arange(anchor_count), type=pa.int64()),
                "episode_index": pa.array(np.zeros(anchor_count), type=pa.int64()),
                "frame_index": pa.array(np.arange(anchor_count), type=pa.int64()),
                "task_index": pa.array(np.full(anchor_count, 7), type=pa.int64()),
                "timestamp": pa.array(
                    np.arange(anchor_count, dtype=np.float32) / 15,
                    type=pa.float32(),
                ),
            }
        ),
        manifest_path,
    )

    anchor_rows = np.arange(anchor_count)
    state_samples = state_values[anchor_rows]
    pose_samples = pose_values[anchor_rows + horizon]
    action_samples = action_values[
        anchor_rows[:, None] + np.arange(horizon)
    ].reshape(-1, 8)
    stats = {
        STATE_KEY: _feature_stats(state_samples),
        ACTION_KEY: _feature_stats(action_samples),
        EE_POSE_KEY: _feature_stats(pose_samples),
    }
    stats_path = root / "meta/stats.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    info = {
        "features": {
            STATE_KEY: {
                "shape": [8],
                "names": {"axes": [*[f"joint_{i}" for i in range(7)], "gripper"]},
            },
            ACTION_KEY: {
                "shape": [8],
                "names": {"axes": [*[f"joint_{i}" for i in range(7)], "gripper"]},
            },
            EE_POSE_KEY: {
                "shape": [7],
                "names": {"axes": ["x", "y", "z", "rx", "ry", "rz", "gripper"]},
            },
        },
        "goal_pose_prior": {
            "horizon": horizon,
            "partial_build": False,
            "anchor_manifest": "valid_anchor_indices.parquet",
            "target_feature": EE_POSE_KEY,
        },
    }
    info_path = root / "meta/info.json"
    info_path.write_text(json.dumps(info), encoding="utf-8")

    provenance = {
        "recipe_version": 3,
        "anchor_count": anchor_count,
        "anchor_definition": {
            "action_window": "[t:t+15]",
            "goal_offset": horizon,
        },
        "config": {
            "horizon": horizon,
            "max_episodes": None,
            "max_files": None,
            "non_idle": {
                "joint_velocity_delta": 1e-3,
                "min_idle_len": 7,
                "min_non_idle_len": 16,
                "trim_last_steps": horizon,
            },
        },
        "files": [
            {
                "path": str(data_path.relative_to(root)),
                "output_rows": length,
            }
        ],
        "artifact_sha256": {
            "valid_anchor_indices.parquet": _sha256(manifest_path),
            "meta/info.json": _sha256(info_path),
            "meta/stats.json": _sha256(stats_path),
        },
    }
    (root / "goal_pose_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )


class FeatureAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.values = np.asarray(
            [
                [-2.0, -1.0],
                [-1.0, -0.5],
                [0.0, 0.0],
                [1.0, 0.5],
                [2.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.meta = {
            "shape": [2],
            "names": {"axes": ["x", "gripper"]},
        }
        self.stats = {
            "count": [5],
            "min": [-2.0, -1.0],
            "max": [2.0, 1.0],
            "q01": [-1.0, -0.8],
            "q99": [1.0, 0.8],
        }

    def test_gripper_name_produces_false_normalization_mask(self) -> None:
        names, mask = normalization_mask_from_feature_meta(self.meta, 2)
        self.assertEqual(names, ["x", "gripper"])
        self.assertEqual(mask.tolist(), [True, False])
        with self.assertRaisesRegex(AuditContractError, "exactly one gripper"):
            normalization_mask_from_feature_meta(
                {"names": {"axes": ["x", "joint"]}}, 2
            )

    def test_clip_fraction_is_reported_per_dimension(self) -> None:
        report = audit_feature_values(self.values, self.stats, self.meta)
        x = report["dimensions"][0]
        self.assertEqual(x["below_q01"], 1)
        self.assertEqual(x["above_q99"], 1)
        self.assertEqual(x["outside"], 2)
        self.assertAlmostEqual(x["outside_fraction"], 0.4)
        self.assertAlmostEqual(x["clamp_loss_fraction"], 0.4)
        self.assertFalse(report["passed"])

    def test_quantile_forward_clamp_inverse_roundtrip(self) -> None:
        report = audit_feature_values(
            self.values,
            self.stats,
            self.meta,
            outside_fraction_limit=0.5,
            xyz_hard_limit=0.5,
            roundtrip_atol=1e-12,
        )
        self.assertTrue(report["passed"], report["errors"])
        self.assertLessEqual(
            report["dimensions"][0]["roundtrip_max_abs_error"], 1e-12
        )
        gripper = report["dimensions"][1]
        self.assertFalse(gripper["normalize"])
        self.assertEqual((gripper["min"], gripper["max"]), (-1.0, 1.0))
        self.assertIsNone(gripper["roundtrip_max_abs_error"])

    def test_invalid_stats_fail_closed(self) -> None:
        bad_count = {**self.stats, "count": [4]}
        with self.assertRaisesRegex(AuditContractError, "expected 5"):
            validate_feature_stats(
                "feature",
                bad_count,
                self.meta,
                expected_dim=2,
                expected_count=5,
            )
        bad_interval = {**self.stats, "q99": [-1.0, 0.8]}
        with self.assertRaisesRegex(AuditContractError, "q99 <= q01"):
            validate_feature_stats(
                "feature",
                bad_interval,
                self.meta,
                expected_dim=2,
                expected_count=5,
            )


class DatasetAuditTest(unittest.TestCase):
    def test_scans_exact_state_action_and_goal_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_full_recipe_fixture(root)
            report = audit_normalization(root, progress_every=0, chunk_anchors=17)
            self.assertTrue(report["passed"], report["errors"])
            self.assertEqual(report["scan"]["anchors_scanned"], 100)
            self.assertEqual(report["features"][STATE_KEY]["scanned_samples"], 100)
            self.assertEqual(report["features"][ACTION_KEY]["scanned_samples"], 1500)
            self.assertEqual(report["features"][EE_POSE_KEY]["scanned_samples"], 100)
            self.assertEqual(
                report["features"][EE_POSE_KEY]["normalization_mask"],
                [True, True, True, True, True, True, False],
            )

    def test_wrong_metadata_stats_count_fails_before_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_full_recipe_fixture(root)
            stats_path = root / "meta/stats.json"
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            stats[STATE_KEY]["count"] = [99]
            stats_path.write_text(json.dumps(stats), encoding="utf-8")
            report = audit_normalization(root, progress_every=0)
            self.assertFalse(report["passed"])
            self.assertEqual(report["scan"]["episodes_scanned"], 0)
            self.assertTrue(
                any(
                    f"{STATE_KEY} stats count is 99, expected 100" in error
                    for error in report["errors"]
                )
            )


if __name__ == "__main__":
    unittest.main()
