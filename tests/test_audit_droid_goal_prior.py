from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.droid_goal_prior.audit_dataset import (
    CAMERA_KEYS,
    _ranges_to_anchor_mask,
    openpi_nonidle_ranges,
    run_audit,
)
from scripts.droid_goal_prior.prepare_dataset import (
    EE_POSE_KEY,
    HORIZON,
    BuildConfig,
    NonIdleConfig,
    prepare_dataset,
)
from test_prepare_droid_goal_prior import _write_fixture


def _add_video_fixture(root: Path) -> None:
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["video_path"] = (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    for camera in CAMERA_KEYS:
        info["features"][camera] = {
            "dtype": "video",
            "shape": [3, 8, 8],
            "names": ["channels", "height", "width"],
        }
        path = root / f"videos/{camera}/chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    info_path.write_text(json.dumps(info), encoding="utf-8")

    metadata_path = root / "meta/episodes/chunk-000/file-000.parquet"
    metadata = pq.read_table(metadata_path)
    starts = pa.array([0.0, 10 / 15], type=pa.float64())
    stops = pa.array([10 / 15, 30 / 15], type=pa.float64())
    for camera in CAMERA_KEYS:
        metadata = metadata.append_column(
            f"videos/{camera}/chunk_index", pa.array([0, 0], type=pa.int64())
        )
        metadata = metadata.append_column(
            f"videos/{camera}/file_index", pa.array([0, 0], type=pa.int64())
        )
        metadata = metadata.append_column(f"videos/{camera}/from_timestamp", starts)
        metadata = metadata.append_column(f"videos/{camera}/to_timestamp", stops)
    pq.write_table(metadata, metadata_path)


def _prepared_fixture(base: Path) -> tuple[Path, Path]:
    source = base / "source"
    output = base / "derived"
    _write_fixture(source)
    _add_video_fixture(source)
    prepare_dataset(
        BuildConfig(
            source_root=str(source),
            output_root=str(output),
            revision="0eabc778f959c54b8c5aa3626cc1128d2d2e54d4",
            max_episodes=None,
            max_files=None,
            video_symlink="relative",
            smoke_anchors=3,
            horizon=HORIZON,
            non_idle=NonIdleConfig(),
        )
    )
    return source, output


class OpenPiParityTest(unittest.TestCase):
    def test_trim_difference_is_exactly_five_end_frames(self) -> None:
        velocities = np.empty((47, 7), dtype=np.float32)
        velocities[:20] = np.arange(20, dtype=np.float32)[:, None] * 0.01
        velocities[20:27] = velocities[19]
        velocities[27:] = 0.20 + np.arange(20, dtype=np.float32)[:, None] * 0.01

        upstream_ranges = openpi_nonidle_ranges(velocities, trim_last_steps=10)
        adapted_ranges = openpi_nonidle_ranges(velocities, trim_last_steps=15)
        self.assertEqual(upstream_ranges, [(0, 10), (27, 37)])
        self.assertEqual(adapted_ranges, [(0, 5), (27, 32)])

        upstream = _ranges_to_anchor_mask(upstream_ranges, len(velocities))
        adapted = _ranges_to_anchor_mask(adapted_ranges, len(velocities))
        self.assertEqual(
            np.flatnonzero(upstream & ~adapted).tolist(),
            [5, 6, 7, 8, 9],
        )

    def test_idle_threshold_has_no_off_by_one(self) -> None:
        velocities = np.arange(40, dtype=np.float32)[:, None] * np.ones((1, 7))
        velocities[10:16] = velocities[9]
        no_split = openpi_nonidle_ranges(velocities, trim_last_steps=10)
        velocities[16] = velocities[9]
        split = openpi_nonidle_ranges(velocities, trim_last_steps=10)
        self.assertEqual(no_split, [(0, 30)])
        self.assertNotEqual(split, no_split)


class AuditIntegrationTest(unittest.TestCase):
    def test_valid_fixture_passes_sampling_and_video_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, output = _prepared_fixture(Path(temporary))
            report = run_audit(
                source,
                output,
                seed=17,
                samples_per_category=8,
                parity_episodes=0,
                manifest_batch_size=2,
            )
            self.assertEqual(report["status"], "passed", report["differences"])
            self.assertEqual(report["manifest"]["rows"], 5)
            self.assertEqual(report["samples"]["unique_anchors"], 5)
            self.assertEqual(report["videos"]["sampled_episodes"], 1)
            self.assertEqual(report["videos"]["inventory_episodes"], 2)
            self.assertEqual(report["videos"]["inventory_missing_files"], 0)

    def test_t_plus_15_pose_off_by_one_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, output = _prepared_fixture(Path(temporary))
            path = output / "data/chunk-000/file-001.parquet"
            table = pq.read_table(path)
            poses = np.asarray(table[EE_POSE_KEY].to_pylist(), dtype=np.float32)
            poses[10, 0] += 1.0  # global index 25 is goal t+15 for anchor 10.
            replacement = pa.FixedSizeListArray.from_arrays(
                pa.array(poses.reshape(-1), type=pa.float32()), 7
            )
            table = table.set_column(
                table.schema.get_field_index(EE_POSE_KEY), EE_POSE_KEY, replacement
            )
            pq.write_table(table, path)

            report = run_audit(
                source,
                output,
                samples_per_category=8,
                parity_episodes=0,
                manifest_batch_size=2,
            )
            categories = {item["category"] for item in report["differences"]}
            self.assertEqual(report["status"], "failed")
            self.assertIn("sample.ee_pose", categories)

    def test_manifest_out_of_order_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, output = _prepared_fixture(Path(temporary))
            path = output / "valid_anchor_indices.parquet"
            table = pq.read_table(path)
            pq.write_table(table.take(pa.array([1, 0, 2, 3, 4])), path)

            report = run_audit(
                source,
                output,
                samples_per_category=8,
                parity_episodes=0,
                manifest_batch_size=2,
            )
            categories = {item["category"] for item in report["differences"]}
            self.assertEqual(report["status"], "failed")
            self.assertIn("manifest.order", categories)


if __name__ == "__main__":
    unittest.main()
