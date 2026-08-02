from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from scripts.droid_goal_prior.prepare_dataset import (
    ACTION_KEY,
    DEFAULT_REVISION,
    EE_POSE_KEY,
    HORIZON,
    STATE_KEY,
    BuildConfig,
    NonIdleConfig,
    _episode_arrays,
    _episode_tables,
    _sha256,
    _verify_pairing_and_poses,
    anchor_mask,
    non_idle_anchor_mask,
    plan_source_files,
    prepare_dataset,
    rpy_to_rotvec,
    verify_dataset,
)


def _write_fixture(root: Path) -> None:
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "videos").mkdir()
    marker = root / ".cache/huggingface/download/meta/info.json.metadata"
    marker.parent.mkdir(parents=True)
    marker.write_text(f"{DEFAULT_REVISION}\nfixture-etag\n0\n", encoding="utf-8")

    rows: list[dict[str, object]] = []
    global_index = 0
    for episode_index, length in ((0, 10), (1, 20)):
        for frame_index in range(length):
            x = episode_index + frame_index * 0.01
            state = np.arange(8, dtype=np.float32) * 0.1 + frame_index * 0.02
            action = state + 0.02
            joint_velocity = np.full(7, frame_index * 0.01, dtype=np.float32)
            rows.append(
                {
                    "extra": f"row-{global_index}",
                    "language_instruction": "move the object",
                    "language_instruction_2": "",
                    "language_instruction_3": "",
                    "observation.state.gripper_position": np.float32(frame_index / 20),
                    "observation.state.cartesian_position": np.asarray(
                        [x, 0.1, 0.2, 0.0, 0.0, frame_index * 0.02],
                        dtype=np.float32,
                    ).tolist(),
                    "observation.state": state.tolist(),
                    "action": action.tolist(),
                    "action.joint_velocity": joint_velocity.tolist(),
                    "is_episode_successful": True,
                    "is_last": frame_index == length - 1,
                    "timestamp": np.float32(frame_index / 15),
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": global_index,
                    "task_index": episode_index + 7,
                }
            )
            global_index += 1

    table = pa.Table.from_pylist(rows)
    # Force the second episode to cross a parquet-file boundary.
    pq.write_table(table.slice(0, 15), root / "data/chunk-000/file-000.parquet")
    pq.write_table(table.slice(15), root / "data/chunk-000/file-001.parquet")
    pq.write_table(
        pa.table({"task_index": pa.array([7, 8]), "task": ["short", "long"]}),
        root / "meta/tasks.parquet",
    )
    episode_meta = root / "meta/episodes/chunk-000/file-000.parquet"
    episode_meta.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0, 1], type=pa.int64()),
                "length": pa.array([10, 20], type=pa.int64()),
                "dataset_from_index": pa.array([0, 10], type=pa.int64()),
                "dataset_to_index": pa.array([10, 30], type=pa.int64()),
                "data/chunk_index": pa.array([0, 0], type=pa.int64()),
                "data/file_index": pa.array([0, 1], type=pa.int64()),
            }
        ),
        episode_meta,
    )
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 15,
                "total_episodes": 2,
                "total_frames": 30,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "features": {
                    STATE_KEY: {"dtype": "float32", "shape": [8]},
                    ACTION_KEY: {"dtype": "float32", "shape": [8]},
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "meta/stats.json").write_text(
        json.dumps({STATE_KEY: {}, ACTION_KEY: {}}), encoding="utf-8"
    )


class RpyConversionTest(unittest.TestCase):
    def test_rpy_matches_scipy_xyz_and_float32(self) -> None:
        rpy = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [np.pi / 2, 0.0, 0.0],
                [0.3, -0.4, 1.2],
                [-2.5, 1.0, -0.7],
            ],
            dtype=np.float64,
        )
        actual = rpy_to_rotvec(rpy)
        expected = Rotation.from_euler("xyz", rpy).as_rotvec().astype(np.float32)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(
            Rotation.from_rotvec(actual).as_matrix(),
            Rotation.from_euler("xyz", rpy).as_matrix(),
            atol=2e-6,
            rtol=0,
        )


class AnchorBoundaryTest(unittest.TestCase):
    def test_openpi_non_idle_segmentation_and_horizon_trim(self) -> None:
        velocities = np.empty((47, 7), dtype=np.float32)
        velocities[:20] = np.arange(20, dtype=np.float32)[:, None] * 0.01
        velocities[20:27] = velocities[19]
        velocities[27:] = 0.20 + np.arange(20, dtype=np.float32)[:, None] * 0.01
        mask = non_idle_anchor_mask(velocities, NonIdleConfig())
        self.assertEqual(np.flatnonzero(mask).tolist(), [0, 1, 2, 3, 4, 27, 28, 29, 30, 31])

    def test_smoke_limits_keep_original_prefix_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            by_episode = plan_source_files(root, max_episodes=1)
            self.assertEqual(
                [(plan.relative_path, plan.rows) for plan in by_episode],
                [("data/chunk-000/file-000.parquet", 10)],
            )
            by_file = plan_source_files(root, max_files=1)
            self.assertEqual(
                [(plan.relative_path, plan.rows) for plan in by_file],
                [("data/chunk-000/file-000.parquet", 10)],
            )

    def test_anchor_never_crosses_episode_or_file_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            plans = [
                type("Plan", (), {"relative_path": "data/chunk-000/file-000.parquet", "rows": 15})(),
                type("Plan", (), {"relative_path": "data/chunk-000/file-001.parquet", "rows": 15})(),
            ]
            columns = [
                "index",
                "episode_index",
                "frame_index",
                "task_index",
                "timestamp",
                "is_episode_successful",
                "language_instruction",
                "language_instruction_2",
                "language_instruction_3",
                STATE_KEY,
                ACTION_KEY,
                "action.joint_velocity",
            ]
            tables = list(_episode_tables(root, plans, columns))
            self.assertEqual([len(table) for table in tables], [10, 20])
            self.assertEqual([table["episode_index"][0].as_py() for table in tables], [0, 1])

            # anchor_mask also needs EE pose; append a deterministic pose for this unit test.
            anchor_counts = []
            for table in tables:
                cart = np.zeros((len(table), 7), dtype=np.float32)
                cart[:, 0] = np.arange(len(table), dtype=np.float32) * 0.01
                table = table.append_column(
                    EE_POSE_KEY,
                    pa.FixedSizeListArray.from_arrays(pa.array(cart.reshape(-1)), 7),
                )
                arrays = _episode_arrays(table)
                mask, windows = anchor_mask(arrays, NonIdleConfig())
                anchor_counts.append(int(mask.sum()))
                self.assertEqual(windows.shape, (max(0, len(table) - HORIZON), HORIZON, 8))
            self.assertEqual(anchor_counts, [0, 5])


class PrepareIntegrationTest(unittest.TestCase):
    def test_pairing_anchor_stats_hash_and_idempotency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            output = base / "derived"
            _write_fixture(source)
            config = BuildConfig(
                source_root=str(source),
                output_root=str(output),
                revision=DEFAULT_REVISION,
                max_episodes=None,
                max_files=None,
                video_symlink="relative",
                smoke_anchors=3,
                horizon=HORIZON,
                non_idle=NonIdleConfig(),
            )
            source_hashes_before = {
                path.relative_to(source): _sha256(path)
                for path in source.rglob("*")
                if path.is_file()
            }
            result = prepare_dataset(config)
            self.assertEqual(result, {"files": 2, "anchors": 5, "successful_episodes": 2})
            self.assertTrue((output / "_SUCCESS.json").is_file())
            self.assertTrue((output / "meta/episodes/chunk-000/file-000.parquet").is_file())
            self.assertEqual(verify_dataset(output), result)
            self.assertEqual(prepare_dataset(config), result)
            source_hashes_after = {
                path.relative_to(source): _sha256(path)
                for path in source.rglob("*")
                if path.is_file()
            }
            self.assertEqual(source_hashes_after, source_hashes_before)

            source_table = pq.read_table(source / "data/chunk-000/file-000.parquet")
            derived_table = pq.read_table(output / "data/chunk-000/file-000.parquet")
            self.assertTrue(derived_table.select(source_table.column_names).equals(source_table))
            self.assertEqual(
                derived_table.schema.field(EE_POSE_KEY).type.value_type, pa.float32()
            )

            anchors = pq.read_table(output / "valid_anchor_indices.parquet").to_pydict()
            self.assertEqual(anchors["episode_index"], [1] * 5)
            self.assertEqual(anchors["frame_index"], [0, 1, 2, 3, 4])
            self.assertEqual(anchors["index"], [10, 11, 12, 13, 14])

            all_derived = pa.concat_tables(
                [
                    pq.read_table(output / "data/chunk-000/file-000.parquet"),
                    pq.read_table(output / "data/chunk-000/file-001.parquet"),
                ]
            )
            state = np.asarray(all_derived[STATE_KEY].to_pylist(), dtype=np.float32)
            goal = np.asarray(all_derived[EE_POSE_KEY].to_pylist(), dtype=np.float32)
            action = np.asarray(all_derived[ACTION_KEY].to_pylist(), dtype=np.float32)
            expected_state = state[10:15]
            expected_goal = goal[25:30]
            expected_action = np.concatenate([action[10 + i : 25 + i] for i in range(5)])
            stats = json.loads((output / "anchor_stats.json").read_text(encoding="utf-8"))
            for key, expected in (
                (STATE_KEY, expected_state),
                (EE_POSE_KEY, expected_goal),
                (ACTION_KEY, expected_action),
            ):
                actual = stats["features"][key]
                np.testing.assert_allclose(actual["min"], expected.min(axis=0))
                np.testing.assert_allclose(actual["max"], expected.max(axis=0))
                np.testing.assert_allclose(actual["mean"], expected.mean(axis=0), rtol=1e-6)
                np.testing.assert_allclose(actual["std"], expected.std(axis=0), rtol=1e-6)
                for quantile in (0.01, 0.10, 0.50, 0.90, 0.99):
                    np.testing.assert_allclose(
                        actual[f"q{int(quantile * 100):02d}"],
                        np.quantile(expected, quantile, axis=0),
                        rtol=1e-6,
                    )

            provenance = json.loads(
                (output / "goal_pose_provenance.json").read_text(encoding="utf-8")
            )
            record = provenance["files"][0]
            self.assertEqual(record["source_sha256"], _sha256(source / record["path"]))
            self.assertEqual(record["output_sha256"], _sha256(output / record["path"]))
            _verify_pairing_and_poses(source, output, [record])

            successful_path = output / "successful_episodes.json"
            successful_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact hash changed"):
                verify_dataset(output)


if __name__ == "__main__":
    unittest.main()
