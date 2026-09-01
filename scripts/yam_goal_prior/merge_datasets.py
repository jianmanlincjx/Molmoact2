#!/usr/bin/env python3
"""Merge the three bimanual YAM task datasets into one LeRobot v3.0 dataset.

``blocks_filtered``, ``dustpan_filtered`` and ``transfer_filtered`` are three
independently recorded datasets that share a byte-identical feature schema but
each number their episodes, rows and tasks from zero. Concatenating them
naively is quietly wrong in five ways, and only one of the five would ever
raise:

1. ``task_index`` is ``0`` in all three despite three different instructions.
   Nothing checks this, so the merged set would train a language-blind policy
   that maps one token sequence to three incompatible behaviours.
2. ``episode_index`` restarts, so episodes from different tasks would collide
   and the per-episode quaternion canonicalization would run across a seam.
3. The global ``index`` restarts, breaking the contiguity that
   ``prepare_dataset.anchor_mask`` asserts.
4. ``data/file_index`` is ``0`` everywhere, so every episode would claim to
   live in ``data/chunk-000/file-000.parquet``.
5. Video ``file_index`` restarts per source, and the shard split points differ
   per camera, so frames would be read from the wrong recording.

This script produces a merged dataset that is a legitimate v3.0 dataset in its
own right, so that ``prepare_dataset.py`` can then run over it unmodified --
keeping the validated canonicalization, anchor and verification code untouched.

Videos are never decoded or re-encoded: each source shard is symlinked under a
remapped ``file_index``, which is exactly why the per-episode
``from_timestamp``/``to_timestamp`` values stay valid (they are relative to
their own shard, and the shard contents do not change).

``meta/stats.json`` is recomputed as a true global statistic rather than
aggregated from per-episode stats. That aggregation is the upstream v3.0 defect
that produced LIBERO v3, and it is live in this data: ``dustpan_filtered``'s
shipped quantiles clip 18.6% of ``action`` ``left_eef.qz`` and
``transfer_filtered``'s clip 13.2% of ``action`` ``left_eef.x``.

Builds happen in a sibling staging directory and are published with a single
``os.replace`` only after a strict verification pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prepare_dataset import (
    CAMERA_KEYS,
    QUANTILES,
    SOURCE_ACTION_KEY,
    SOURCE_STATE_KEY,
    _json_dump,
    _sha256,
    source_content_digest,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_NAMES = ("blocks_filtered", "dustpan_filtered", "transfer_filtered")
DEFAULT_DATASET_ROOT = _REPO_ROOT / "real_robot_datasets"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATASET_ROOT / "yam_3task"
RECIPE_VERSION = 1

EXPECTED_FPS = 30
EXPECTED_ROBOT_TYPE = "bi_yam_follower"
EXPECTED_CODEBASE_VERSION = "v3.0"

# Columns rewritten during the merge. Everything else is carried through
# byte-for-byte, including the source columns prepare_dataset.py later drops.
REMAPPED_DATA_COLUMNS = ("episode_index", "index", "task_index")
# Episode-metadata columns that carry an index into a file or the global row
# space, and therefore have to be rebuilt rather than concatenated.
REMAPPED_EPISODE_COLUMNS = (
    "episode_index",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "dataset_to_index",
    "meta/episodes/chunk_index",
    "meta/episodes/file_index",
)
IMAGE_STAT_KEYS = tuple(CAMERA_KEYS)


@dataclass(frozen=True)
class MergeConfig:
    dataset_root: str
    output_root: str
    sources: tuple[str, ...]
    video_link_mode: str = "relative"
    source_digests: dict[str, str] = field(default_factory=dict)


def _config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"recipe_version": RECIPE_VERSION, "config": config},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


# --------------------------------------------------------------------------
# Source inspection
# --------------------------------------------------------------------------


def _read_info(root: Path) -> dict[str, Any]:
    return json.loads((root / "meta/info.json").read_text(encoding="utf-8"))


def _data_parquet(root: Path) -> Path:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if len(files) != 1:
        raise ValueError(
            f"{root} has {len(files)} data parquet files; the merge assumes a single "
            "file per source (all three sources ship one)"
        )
    return files[0]


def _episode_parquet(root: Path) -> Path:
    files = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    if len(files) != 1:
        raise ValueError(f"{root} has {len(files)} episode-metadata files; expected 1")
    return files[0]


def _task_text(root: Path) -> str:
    table = pq.read_table(root / "meta/tasks.parquet")
    text_key = "task" if "task" in table.column_names else "__index_level_0__"
    texts = table.column(text_key).to_pylist()
    indices = table.column("task_index").to_pylist()
    if len(texts) != 1 or int(indices[0]) != 0:
        raise ValueError(
            f"{root} has {len(texts)} tasks with indices {indices}; the merge assumes "
            "each source contributes exactly one task numbered 0"
        )
    text = str(texts[0]).strip()
    if not text:
        raise ValueError(f"{root} has an empty task string")
    return text


def _check_schema_agreement(roots: list[Path]) -> dict[str, Any]:
    """Refuse to merge sources that do not share one feature contract."""
    reference = _read_info(roots[0])
    for root in roots:
        info = _read_info(root)
        for key in ("codebase_version", "robot_type", "fps", "chunks_size", "data_path", "video_path"):
            if info[key] != reference[key]:
                raise ValueError(
                    f"{root.name} has {key}={info[key]!r}, {roots[0].name} has "
                    f"{reference[key]!r}; sources must agree"
                )
        if info["features"] != reference["features"]:
            differing = sorted(
                k
                for k in set(info["features"]) | set(reference["features"])
                if info["features"].get(k) != reference["features"].get(k)
            )
            raise ValueError(f"{root.name} feature schema differs from {roots[0].name}: {differing}")
    if reference["fps"] != EXPECTED_FPS:
        raise ValueError(f"expected {EXPECTED_FPS} fps, got {reference['fps']}")
    if reference["robot_type"] != EXPECTED_ROBOT_TYPE:
        raise ValueError(f"expected robot_type {EXPECTED_ROBOT_TYPE}, got {reference['robot_type']}")
    if reference["codebase_version"] != EXPECTED_CODEBASE_VERSION:
        raise ValueError(f"expected {EXPECTED_CODEBASE_VERSION}, got {reference['codebase_version']}")
    for key in (SOURCE_STATE_KEY, SOURCE_ACTION_KEY):
        if key not in reference["features"]:
            raise ValueError(f"sources are missing {key}")
    for key in CAMERA_KEYS:
        if key not in reference["features"]:
            raise ValueError(f"sources are missing camera {key}")
    return reference


def _video_shards(root: Path, camera: str) -> list[Path]:
    return sorted((root / "videos" / camera).glob("chunk-*/*.mp4"))


def _video_file_indices(episodes: pa.Table, camera: str) -> np.ndarray:
    return np.asarray(episodes.column(f"videos/{camera}/file_index").to_numpy(zero_copy_only=False))


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------


def _remap_data_table(
    table: pa.Table, episode_offset: int, row_offset: int, task_index: int
) -> pa.Table:
    rows = table.num_rows
    episode = np.asarray(table.column("episode_index").to_numpy(zero_copy_only=False))
    replacements = {
        "episode_index": pa.array(episode + episode_offset, type=pa.int64()),
        "index": pa.array(np.arange(row_offset, row_offset + rows, dtype=np.int64), type=pa.int64()),
        "task_index": pa.array(np.full(rows, task_index, dtype=np.int64), type=pa.int64()),
    }
    for name, column in replacements.items():
        table = table.set_column(table.schema.get_field_index(name), name, column)
    return table


def _remap_episode_table(
    episodes: pa.Table,
    episode_offset: int,
    row_offset: int,
    video_offsets: dict[str, int],
) -> pa.Table:
    n = episodes.num_rows
    lengths = np.asarray(episodes.column("length").to_numpy(zero_copy_only=False))
    starts = row_offset + np.concatenate(([0], np.cumsum(lengths)[:-1]))
    updates: dict[str, pa.Array] = {
        "episode_index": pa.array(
            np.asarray(episodes.column("episode_index").to_numpy(zero_copy_only=False))
            + episode_offset,
            type=pa.int64(),
        ),
        # The merged build writes one data file and one episode-metadata file.
        "data/chunk_index": pa.array(np.zeros(n, dtype=np.int64), type=pa.int64()),
        "data/file_index": pa.array(np.zeros(n, dtype=np.int64), type=pa.int64()),
        "meta/episodes/chunk_index": pa.array(np.zeros(n, dtype=np.int64), type=pa.int64()),
        "meta/episodes/file_index": pa.array(np.zeros(n, dtype=np.int64), type=pa.int64()),
        "dataset_from_index": pa.array(starts, type=pa.int64()),
        "dataset_to_index": pa.array(starts + lengths, type=pa.int64()),
    }
    for camera, offset in video_offsets.items():
        key = f"videos/{camera}/file_index"
        updates[key] = pa.array(_video_file_indices(episodes, camera) + offset, type=pa.int64())
    for name, column in updates.items():
        episodes = episodes.set_column(episodes.schema.get_field_index(name), name, column)
    return episodes


def _align_columns(tables: list[pa.Table], label: str) -> list[pa.Table]:
    """Reorder tables onto the first table's column order.

    The three sources were written independently and their episode-metadata
    parquet files list the same ~150 columns in a different (hash-dependent)
    order, which ``pa.concat_tables`` rejects. The column *set* and the column
    *types* must still agree exactly -- a difference there is a real schema
    divergence, not a cosmetic one, so it raises.
    """
    reference = tables[0].schema
    order = [field.name for field in reference]
    aligned = [tables[0]]
    for position, table in enumerate(tables[1:], start=1):
        if set(table.column_names) != set(order):
            difference = sorted(set(table.column_names) ^ set(order))
            raise ValueError(f"{label} source {position} column set differs: {difference}")
        table = table.select(order)
        for field, other in zip(reference, table.schema, strict=True):
            if field.type != other.type:
                raise ValueError(
                    f"{label} source {position} column {field.name} is {other.type}, "
                    f"expected {field.type}"
                )
        aligned.append(table)
    return aligned


def _link_videos(
    source_roots: list[Path],
    video_offsets: list[dict[str, int]],
    staging_root: Path,
    final_root: Path,
    mode: str,
) -> list[dict[str, str]]:
    """Symlink every source video shard under its remapped ``file_index``."""
    links: list[dict[str, str]] = []
    for camera in CAMERA_KEYS:
        destination_dir = staging_root / "videos" / camera / "chunk-000"
        destination_dir.mkdir(parents=True, exist_ok=True)
        for root, offsets in zip(source_roots, video_offsets, strict=True):
            for shard in _video_shards(root, camera):
                file_index = int(shard.stem.split("-")[-1]) + offsets[camera]
                link = destination_dir / f"file-{file_index:03d}.mp4"
                if link.exists() or link.is_symlink():
                    raise ValueError(f"video file_index collision at {link}")
                final_dir = final_root / "videos" / camera / "chunk-000"
                if mode == "absolute":
                    target = str(shard.resolve())
                elif mode == "relative":
                    target = os.path.relpath(shard.resolve(), start=final_dir.resolve())
                else:
                    raise ValueError(f"unknown video symlink mode: {mode}")
                link.symlink_to(target)
                links.append(
                    {
                        "link": str(link.relative_to(staging_root)),
                        "target": target,
                        "source": str(shard.resolve()),
                    }
                )
    return links


def _stat_block(values: np.ndarray, template: Any) -> dict[str, Any]:
    """Exact global statistics for one column, shaped like the source entry."""
    if values.ndim == 1:
        values = values[:, None]
    shape = np.asarray(template["mean"], dtype=np.float64).shape
    work = values.astype(np.float64, copy=False)

    def _shape(array: np.ndarray) -> Any:
        return np.asarray(array, dtype=np.float64).reshape(shape).tolist()

    block = {
        "count": [int(len(work))],
        "mean": _shape(work.mean(axis=0)),
        "std": _shape(work.std(axis=0)),
        "min": _shape(work.min(axis=0)),
        "max": _shape(work.max(axis=0)),
    }
    for quantile in QUANTILES:
        block[f"q{int(round(quantile * 100)):02d}"] = _shape(
            np.quantile(work, quantile, axis=0, method="linear")
        )
    return block


def _aggregate_image_stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Count-weighted aggregation for image channels.

    Image statistics are sampled from decoded frames, not stored in the parquet,
    so recomputing them exactly would mean decoding 4.7 GB of AV1 for values the
    MolmoAct2 processor does not use for normalization. Mean, std, min and max
    aggregate exactly; the quantiles are count-weighted averages and are
    therefore approximate, which is recorded in the provenance.
    """
    counts = np.array([float(np.asarray(e["count"]).ravel()[0]) for e in entries])
    total = counts.sum()
    weights = counts / total
    shape = np.asarray(entries[0]["mean"], dtype=np.float64).shape
    means = np.stack([np.asarray(e["mean"], dtype=np.float64) for e in entries])
    stds = np.stack([np.asarray(e["std"], dtype=np.float64) for e in entries])
    w = weights.reshape((-1,) + (1,) * len(shape))
    mean = (means * w).sum(axis=0)
    second = ((stds**2 + means**2) * w).sum(axis=0)
    block = {
        "count": [int(total)],
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(second - mean**2, 0.0)).tolist(),
        "min": np.stack([np.asarray(e["min"], dtype=np.float64) for e in entries])
        .min(axis=0)
        .tolist(),
        "max": np.stack([np.asarray(e["max"], dtype=np.float64) for e in entries])
        .max(axis=0)
        .tolist(),
    }
    for quantile in QUANTILES:
        key = f"q{int(round(quantile * 100)):02d}"
        stacked = np.stack([np.asarray(e[key], dtype=np.float64) for e in entries])
        block[key] = (stacked * w).sum(axis=0).tolist()
    return block


def _build_stats(table: pa.Table, source_stats: list[dict[str, Any]]) -> dict[str, Any]:
    reference = source_stats[0]
    merged: dict[str, Any] = {}
    for key in sorted(reference):
        if key in IMAGE_STAT_KEYS:
            merged[key] = _aggregate_image_stats([stats[key] for stats in source_stats])
            continue
        if key not in table.column_names:
            raise ValueError(f"stats.json has {key} but the merged table does not")
        column = table.column(key)
        if pa.types.is_fixed_size_list(column.type) or pa.types.is_list(column.type):
            values = np.stack(column.to_numpy(zero_copy_only=False))
        else:
            values = np.asarray(column.to_numpy(zero_copy_only=False))
        merged[key] = _stat_block(values, reference[key])
    return merged


def _merged_tasks_table(texts: list[str]) -> pa.Table:
    if len(set(texts)) != len(texts):
        raise ValueError(f"sources share a task string: {texts}")
    frame = pd.DataFrame({"task_index": list(range(len(texts)))}, index=pd.Index(texts))
    return pa.Table.from_pandas(frame, preserve_index=True)


def merge_datasets(config: MergeConfig) -> dict[str, Any]:
    dataset_root = Path(config.dataset_root)
    output_root = Path(config.output_root)
    source_roots = [dataset_root / name for name in config.sources]
    for root in source_roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
    if len(set(config.sources)) != len(config.sources):
        raise ValueError("duplicate source datasets requested")
    if output_root.exists():
        raise FileExistsError(f"{output_root} already exists; remove it or pick another --output-root")

    reference_info = _check_schema_agreement(source_roots)
    digests = {root.name: source_content_digest(root) for root in source_roots}
    fingerprint = _config_fingerprint({**asdict(config), "source_digests": digests})

    staging_root = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    shutil.rmtree(staging_root, ignore_errors=True)
    staging_root.mkdir(parents=True)
    try:
        data_tables: list[pa.Table] = []
        episode_tables: list[pa.Table] = []
        source_stats: list[dict[str, Any]] = []
        task_texts: list[str] = []
        summaries: list[dict[str, Any]] = []
        video_offsets: list[dict[str, int]] = []

        episode_offset = 0
        row_offset = 0
        camera_cursor = {camera: 0 for camera in CAMERA_KEYS}

        for task_index, root in enumerate(source_roots):
            info = _read_info(root)
            data = pq.read_table(_data_parquet(root))
            episodes = pq.read_table(_episode_parquet(root))
            if data.num_rows != info["total_frames"]:
                raise ValueError(f"{root.name}: {data.num_rows} rows vs info {info['total_frames']}")
            if episodes.num_rows != info["total_episodes"]:
                raise ValueError(
                    f"{root.name}: {episodes.num_rows} episodes vs info {info['total_episodes']}"
                )
            offsets = dict(camera_cursor)
            video_offsets.append(offsets)

            data_tables.append(_remap_data_table(data, episode_offset, row_offset, task_index))
            episode_tables.append(_remap_episode_table(episodes, episode_offset, row_offset, offsets))
            source_stats.append(json.loads((root / "meta/stats.json").read_text(encoding="utf-8")))
            task_texts.append(_task_text(root))
            summaries.append(
                {
                    "name": root.name,
                    "task_index": task_index,
                    "task": task_texts[-1],
                    "episodes": episodes.num_rows,
                    "frames": data.num_rows,
                    "episode_offset": episode_offset,
                    "row_offset": row_offset,
                    "video_file_index_offsets": offsets,
                    "content_digest": digests[root.name],
                }
            )
            for camera in CAMERA_KEYS:
                shards = len(_video_shards(root, camera))
                if shards == 0:
                    raise ValueError(f"{root.name} has no {camera} video shards")
                observed = set(int(i) for i in _video_file_indices(episodes, camera))
                if observed != set(range(shards)):
                    raise ValueError(
                        f"{root.name}/{camera}: episode metadata references file_index "
                        f"{sorted(observed)} but {shards} shards exist on disk"
                    )
                camera_cursor[camera] += shards
            episode_offset += episodes.num_rows
            row_offset += data.num_rows

        merged_data = pa.concat_tables(_align_columns(data_tables, "data"))
        merged_episodes = pa.concat_tables(_align_columns(episode_tables, "episode metadata"))

        (staging_root / "data/chunk-000").mkdir(parents=True)
        (staging_root / "meta/episodes/chunk-000").mkdir(parents=True)
        pq.write_table(merged_data, staging_root / "data/chunk-000/file-000.parquet")
        pq.write_table(merged_episodes, staging_root / "meta/episodes/chunk-000/file-000.parquet")
        pq.write_table(_merged_tasks_table(task_texts), staging_root / "meta/tasks.parquet")

        info = dict(reference_info)
        info["total_episodes"] = merged_episodes.num_rows
        info["total_frames"] = merged_data.num_rows
        info["total_tasks"] = len(task_texts)
        info["splits"] = {"train": f"0:{merged_episodes.num_rows}"}
        _json_dump(staging_root / "meta/info.json", info)
        _json_dump(staging_root / "meta/stats.json", _build_stats(merged_data, source_stats))

        links = _link_videos(
            source_roots, video_offsets, staging_root, output_root, config.video_link_mode
        )

        provenance = {
            "recipe_version": RECIPE_VERSION,
            "config_fingerprint": fingerprint,
            "config": asdict(config),
            "sources": summaries,
            "total_episodes": merged_episodes.num_rows,
            "total_frames": merged_data.num_rows,
            "total_tasks": len(task_texts),
            "remapped_data_columns": list(REMAPPED_DATA_COLUMNS),
            "remapped_episode_columns": list(REMAPPED_EPISODE_COLUMNS),
            "task_assignment": {text: i for i, text in enumerate(task_texts)},
            "statistics": {
                "scope": "all frames of all merged episodes",
                "method": "exact global statistics recomputed from the merged parquet",
                "reason": (
                    "LeRobot v3.0 aggregates per-episode stats, which is not the true "
                    "global quantile; on these sources that clips up to 24% of a single "
                    "dimension"
                ),
                "image_stats": (
                    "count-weighted aggregation of the source values; mean/std/min/max "
                    "are exact, quantiles are approximate"
                ),
            },
            "videos": {
                "mode": config.video_link_mode,
                "decoded": False,
                "note": (
                    "per-episode from_timestamp/to_timestamp are relative to their own "
                    "shard, so remapping file_index without touching shard contents "
                    "keeps them valid"
                ),
                "links": links,
            },
            "artifact_sha256": {
                str(path.relative_to(staging_root)): _sha256(path)
                for path in sorted(staging_root.rglob("*"))
                if path.is_file() and not path.is_symlink()
            },
        }
        _json_dump(staging_root / "merge_provenance.json", provenance)
        result = verify_merged(staging_root, dataset_root=dataset_root, require_success=False)
        _json_dump(
            staging_root / "_SUCCESS.json",
            {
                "status": "complete",
                "config_fingerprint": fingerprint,
                "provenance_sha256": _sha256(staging_root / "merge_provenance.json"),
            },
        )
        os.replace(staging_root, output_root)
        return result
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def verify_merged(
    root: Path, dataset_root: Path | None = None, require_success: bool = True
) -> dict[str, Any]:
    """Re-derive the merged dataset from its sources and compare, row for row."""
    root = Path(root)
    provenance = json.loads((root / "merge_provenance.json").read_text(encoding="utf-8"))
    if require_success:
        success = json.loads((root / "_SUCCESS.json").read_text(encoding="utf-8"))
        if success.get("status") != "complete":
            raise ValueError(f"{root} is not a completed build")
        if success.get("provenance_sha256") != _sha256(root / "merge_provenance.json"):
            raise ValueError("merge_provenance.json changed after the build")

    for relative, expected in provenance["artifact_sha256"].items():
        actual = _sha256(root / relative)
        if actual != expected:
            raise ValueError(f"{relative} hash is {actual}, expected {expected}")

    dataset_root = Path(dataset_root) if dataset_root else Path(provenance["config"]["dataset_root"])
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    data = pq.read_table(root / "data/chunk-000/file-000.parquet")
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet")
    tasks = pq.read_table(root / "meta/tasks.parquet")

    checks: dict[str, Any] = {}

    # -- shape and identity columns -----------------------------------------
    if data.num_rows != info["total_frames"]:
        raise ValueError(f"data has {data.num_rows} rows, info says {info['total_frames']}")
    if episodes.num_rows != info["total_episodes"]:
        raise ValueError(f"{episodes.num_rows} episodes, info says {info['total_episodes']}")
    if info["total_tasks"] != tasks.num_rows:
        raise ValueError(f"info says {info['total_tasks']} tasks, tasks.parquet has {tasks.num_rows}")
    if info["splits"] != {"train": f"0:{episodes.num_rows}"}:
        raise ValueError(f"splits {info['splits']} does not cover all episodes")

    index = np.asarray(data.column("index").to_numpy(zero_copy_only=False))
    if not np.array_equal(index, np.arange(data.num_rows)):
        raise ValueError("global index is not contiguous 0..N-1")
    episode_column = np.asarray(data.column("episode_index").to_numpy(zero_copy_only=False))
    episode_ids = np.asarray(episodes.column("episode_index").to_numpy(zero_copy_only=False))
    if not np.array_equal(episode_ids, np.arange(episodes.num_rows)):
        raise ValueError("episode_index is not contiguous 0..E-1")
    if not np.array_equal(np.unique(episode_column), episode_ids):
        raise ValueError("data and episode metadata disagree on the episode set")
    if not np.all(np.diff(episode_column) >= 0):
        raise ValueError("rows are not grouped by episode")
    checks["rows"] = int(data.num_rows)
    checks["episodes"] = int(episodes.num_rows)

    # -- per-episode row ranges ---------------------------------------------
    lengths = np.asarray(episodes.column("length").to_numpy(zero_copy_only=False))
    starts = np.asarray(episodes.column("dataset_from_index").to_numpy(zero_copy_only=False))
    stops = np.asarray(episodes.column("dataset_to_index").to_numpy(zero_copy_only=False))
    expected_starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    if not np.array_equal(starts, expected_starts) or not np.array_equal(stops, starts + lengths):
        raise ValueError("dataset_from_index/dataset_to_index are not a contiguous cover")
    if int(stops[-1]) != data.num_rows:
        raise ValueError(f"episode ranges cover {stops[-1]} rows, data has {data.num_rows}")
    counts = np.bincount(episode_column, minlength=episodes.num_rows)
    if not np.array_equal(counts, lengths):
        raise ValueError("per-episode row counts do not match the metadata lengths")
    frame_index = np.asarray(data.column("frame_index").to_numpy(zero_copy_only=False))
    for start, stop in zip(starts, stops, strict=True):
        if not np.array_equal(frame_index[start:stop], np.arange(stop - start)):
            raise ValueError(f"frame_index does not restart at 0 for rows [{start}, {stop})")

    # -- task assignment -----------------------------------------------------
    text_key = "task" if "task" in tasks.column_names else "__index_level_0__"
    task_texts = [str(t) for t in tasks.column(text_key).to_pylist()]
    task_indices = [int(i) for i in tasks.column("task_index").to_pylist()]
    if task_indices != list(range(len(task_indices))):
        raise ValueError(f"task_index is not 0..T-1: {task_indices}")
    if len(set(task_texts)) != len(task_texts):
        raise ValueError("merged tasks.parquet contains duplicate instructions")
    task_column = np.asarray(data.column("task_index").to_numpy(zero_copy_only=False))
    if set(np.unique(task_column).tolist()) != set(task_indices):
        raise ValueError("data references task indices absent from tasks.parquet")
    episode_tasks = [list(t) for t in episodes.column("tasks").to_pylist()]
    for episode_id, start, stop in zip(episode_ids, starts, stops, strict=True):
        window = task_column[start:stop]
        if not np.all(window == window[0]):
            raise ValueError(f"task_index changes inside episode {episode_id}")
        expected_text = task_texts[int(window[0])]
        if episode_tasks[int(episode_id)] != [expected_text]:
            raise ValueError(
                f"episode {episode_id} carries tasks {episode_tasks[int(episode_id)]} but "
                f"task_index {int(window[0])} means {expected_text!r}"
            )
    checks["tasks"] = {text: index for text, index in zip(task_texts, task_indices, strict=True)}

    # -- payload equality against the sources --------------------------------
    payload_columns = [c for c in data.column_names if c not in REMAPPED_DATA_COLUMNS]
    row_cursor = 0
    episode_cursor = 0
    for summary in provenance["sources"]:
        source_root = dataset_root / summary["name"]
        actual_digest = source_content_digest(source_root)
        if actual_digest != summary["content_digest"]:
            raise ValueError(
                f"{summary['name']} content digest is {actual_digest}, build recorded "
                f"{summary['content_digest']}; the source changed since the merge"
            )
        source_data = pq.read_table(_data_parquet(source_root))
        rows = source_data.num_rows
        if rows != summary["frames"]:
            raise ValueError(f"{summary['name']} row count changed")
        window = data.slice(row_cursor, rows)
        for column in payload_columns:
            if window.column(column) != source_data.column(column):
                raise ValueError(f"{summary['name']}: merged column {column} differs from source")
        merged_episode = np.asarray(window.column("episode_index").to_numpy(zero_copy_only=False))
        source_episode = np.asarray(
            source_data.column("episode_index").to_numpy(zero_copy_only=False)
        )
        if not np.array_equal(merged_episode, source_episode + summary["episode_offset"]):
            raise ValueError(f"{summary['name']}: episode_index remap is wrong")
        if not np.all(task_column[row_cursor : row_cursor + rows] == summary["task_index"]):
            raise ValueError(f"{summary['name']}: task_index is not {summary['task_index']}")

        # -- video shard remap ------------------------------------------------
        source_episodes = pq.read_table(_episode_parquet(source_root))
        merged_window = episodes.slice(episode_cursor, source_episodes.num_rows)
        for camera in CAMERA_KEYS:
            offset = summary["video_file_index_offsets"][camera]
            expected = _video_file_indices(source_episodes, camera) + offset
            if not np.array_equal(_video_file_indices(merged_window, camera), expected):
                raise ValueError(f"{summary['name']}/{camera}: file_index remap is wrong")
            for key in ("from_timestamp", "to_timestamp"):
                column = f"videos/{camera}/{key}"
                if merged_window.column(column) != source_episodes.column(column):
                    raise ValueError(f"{summary['name']}/{camera}: {key} was altered")
            for shard in _video_shards(source_root, camera):
                file_index = int(shard.stem.split("-")[-1]) + offset
                link = root / f"videos/{camera}/chunk-000/file-{file_index:03d}.mp4"
                if not link.is_symlink():
                    raise ValueError(f"{link} is not a symlink")
                if link.resolve() != shard.resolve():
                    raise ValueError(f"{link} resolves to {link.resolve()}, expected {shard}")
        row_cursor += rows
        episode_cursor += source_episodes.num_rows
    if row_cursor != data.num_rows or episode_cursor != episodes.num_rows:
        raise ValueError("sources do not account for every merged row/episode")

    # -- no stray video shards ----------------------------------------------
    for camera in CAMERA_KEYS:
        links = sorted((root / "videos" / camera / "chunk-000").glob("*.mp4"))
        observed = {int(p.stem.split("-")[-1]) for p in links}
        if observed != set(range(len(links))):
            raise ValueError(f"{camera}: video file_index values {sorted(observed)} are not 0..N-1")
        referenced = set(int(i) for i in _video_file_indices(episodes, camera))
        if referenced != observed:
            raise ValueError(
                f"{camera}: episodes reference {sorted(referenced)}, disk has {sorted(observed)}"
            )
    checks["video_shards"] = {
        camera: len(list((root / "videos" / camera / "chunk-000").glob("*.mp4")))
        for camera in CAMERA_KEYS
    }

    # -- statistics ----------------------------------------------------------
    stored = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
    recomputed = _build_stats(
        data,
        [
            json.loads((dataset_root / s["name"] / "meta/stats.json").read_text(encoding="utf-8"))
            for s in provenance["sources"]
        ],
    )
    if set(stored) != set(recomputed):
        raise ValueError(f"stats keys differ: {sorted(set(stored) ^ set(recomputed))}")
    for key in stored:
        for statistic in recomputed[key]:
            a = np.asarray(stored[key][statistic], dtype=np.float64)
            b = np.asarray(recomputed[key][statistic], dtype=np.float64)
            if not np.allclose(a, b, rtol=0, atol=1e-7):
                raise ValueError(f"stats[{key}][{statistic}] does not reproduce")

    # -- the defect this merge exists to avoid -------------------------------
    clipping: dict[str, float] = {}
    for key in (SOURCE_STATE_KEY, SOURCE_ACTION_KEY):
        values = np.stack(data.column(key).to_numpy(zero_copy_only=False)).astype(np.float64)
        q01 = np.asarray(stored[key]["q01"], dtype=np.float64).ravel()
        q99 = np.asarray(stored[key]["q99"], dtype=np.float64).ravel()
        if not np.all(q99 > q01):
            raise ValueError(f"{key}: q99 is not strictly above q01 in every dimension")
        outside = ((values < q01) | (values > q99)).mean(axis=0)
        clipping[key] = float(outside.max())
        if outside.max() > 0.05:
            raise ValueError(
                f"{key}: worst-dimension clipping is {outside.max():.1%}, above the 5% gate; "
                "the merged statistics are wrong, not the data"
            )
    checks["worst_dim_clipping"] = clipping
    return checks


# --------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SOURCE_NAMES),
        help="source dataset directory names, in the order that assigns task_index",
    )
    parser.add_argument("--video-link-mode", choices=("relative", "absolute"), default="relative")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing merged dataset instead of building one",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verify_only:
        result = verify_merged(args.output_root, dataset_root=args.dataset_root)
        print(json.dumps({"verified": str(args.output_root), **result}, indent=2))
        return
    config = MergeConfig(
        dataset_root=str(args.dataset_root),
        output_root=str(args.output_root),
        sources=tuple(args.sources),
        video_link_mode=args.video_link_mode,
    )
    result = merge_datasets(config)
    print(json.dumps({"merged": str(args.output_root), **result}, indent=2))


if __name__ == "__main__":
    main()
