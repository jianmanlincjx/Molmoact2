#!/usr/bin/env python3
"""Prepare a read-only bimanual YAM v3 dataset for the 30-step goal-pose prior.

The source records absolute end-effector pose for both observation and action.
This build turns that into the canonical LeRobot/MolmoAct2 contract:

* ``observation.state`` and ``action`` are 16-D absolute EEF, per arm
  ``[xyz(3), quat(4, w-first), gripper(1)]``;
* the goal target is simply ``observation.state`` at ``t + 30`` -- the LIBERO
  arrangement, so there is no separate goal feature and no forward kinematics;
* quaternion signs are canonicalized per episode, because ``q`` and ``-q`` are
  the same rotation but differ by ``|dq| = 2`` in the action space, which would
  otherwise inject ~700x-normal jumps straight into the flow-matching loss.

Renaming is not cosmetic. LeRobot maps *every* feature key starting with
``action`` to a policy action feature, so the source's ``action_eef_absolute``,
``action_eef_delta`` and ``action_joint_angles`` would collide. The derived view
keeps only the canonical pair plus identity columns.

``action_eef_delta`` is dropped by design: it is a componentwise subtraction, so
its quaternion block is not a rotation difference and is not usable as-is.

Videos are linked, never decoded. Builds happen in a sibling staging directory
and are published only after a strict verification pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = _REPO_ROOT / "real_robot_datasets/yam_3task"
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "real_robot_datasets/yam_3task_goal_pose"
RECIPE_VERSION = 2
HORIZON = 30
EXPECTED_FPS = 30

STATE_DIM = 16
ACTION_DIM = 16

# Canonical keys the policy consumes.
STATE_KEY = "observation.state"
ACTION_KEY = "action"
# Source keys they are built from.
SOURCE_STATE_KEY = "observation.state_eef_absolute"
SOURCE_ACTION_KEY = "action_eef_absolute"
# Source columns deliberately excluded from the derived view.
DROPPED_SOURCE_KEYS = (
    "action_eef_delta",
    "action_joint_angles",
    "observation.state_joint_angles",
)

# Per-arm layout within the 16-D vector.
ARM_OFFSETS = (0, 8)
QUAT_SLICES = tuple(slice(off + 3, off + 7) for off in ARM_OFFSETS)
FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{side}_eef.{component}"
    for side in ("left", "right")
    for component in ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")
)

CAMERA_KEYS = (
    "observation.images.top",
    "observation.images.left",
    "observation.images.right",
)
IDENTITY_KEYS = (
    "index",
    "episode_index",
    "frame_index",
    "task_index",
    "timestamp",
)
ANCHOR_SCHEMA = pa.schema(
    [
        pa.field("index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("frame_index", pa.int64()),
        pa.field("task_index", pa.int64()),
        pa.field("timestamp", pa.float32()),
    ]
)
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


@dataclass(frozen=True)
class NonIdleConfig:
    """Idle segmentation parameters, rescaled from DROID's 15 Hz to 30 Hz.

    DROID used ``joint_velocity_delta=1e-3`` on ``action.joint_velocity`` with
    ``min_idle_len=7`` (0.47 s at 15 Hz). At 30 Hz a per-step command delta is
    about half as large, so the threshold halves and the idle-run length doubles
    to preserve the same wall-clock semantics. Measured on this data the filter
    removes nothing; it is kept for contract symmetry.
    """

    action_delta: float = 5e-4
    min_idle_len: int = 14
    min_non_idle_len: int = HORIZON + 1
    trim_last_steps: int = HORIZON


@dataclass(frozen=True)
class BuildConfig:
    source_root: str
    output_root: str
    source_digest: str
    max_episodes: int | None
    exclude_episodes: tuple[int, ...]
    video_symlink: str
    smoke_anchors: int
    horizon: int
    non_idle: NonIdleConfig


@dataclass(frozen=True)
class FilePlan:
    relative_path: str
    rows: int


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _config_fingerprint(config: BuildConfig) -> str:
    return _config_dict_fingerprint(asdict(config))


def _config_dict_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"recipe_version": RECIPE_VERSION, "config": config},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_config(config: BuildConfig) -> None:
    if config.horizon != HORIZON:
        raise ValueError(f"horizon is fixed at {HORIZON}")
    if config.smoke_anchors < 0:
        raise ValueError("--smoke-anchors must be non-negative")
    if not np.isfinite(config.non_idle.action_delta) or config.non_idle.action_delta < 0:
        raise ValueError("action delta threshold must be finite and non-negative")
    if config.non_idle.min_idle_len < 1 or config.non_idle.min_non_idle_len < 1:
        raise ValueError("non-idle segment lengths must be positive")
    if config.non_idle.trim_last_steps != config.horizon:
        raise ValueError("non-idle ranges must trim exactly one action/goal horizon")
    if config.non_idle.min_non_idle_len <= config.horizon:
        raise ValueError("a retained run must be longer than one horizon")
    if len(set(config.exclude_episodes)) != len(config.exclude_episodes):
        raise ValueError("--exclude-episodes contains duplicates")
    if any(episode < 0 for episode in config.exclude_episodes):
        raise ValueError("--exclude-episodes must be non-negative")


def _vector_column(table: pa.Table, key: str, width: int) -> np.ndarray:
    column = table.column(key).combine_chunks()
    if pa.types.is_fixed_size_list(column.type):
        if column.type.list_size != width:
            raise ValueError(f"{key} has width {column.type.list_size}, expected {width}")
        values = column.values.to_numpy(zero_copy_only=False)
        array = values.reshape(len(column), width)
    elif pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        offsets = column.offsets.to_numpy(zero_copy_only=False)
        lengths = np.diff(offsets)
        if not np.all(lengths == width):
            raise ValueError(f"{key} has non-{width} rows")
        values = column.values.to_numpy(zero_copy_only=False)
        array = values.reshape(len(column), width)
    else:
        raise TypeError(f"{key} must be a list column, got {column.type}")
    if array.shape != (len(table), width):
        raise ValueError(f"{key} shape is {array.shape}, expected {(len(table), width)}")
    return np.asarray(array)


# --------------------------------------------------------------------------
# Quaternion canonicalization and canonical column construction
# --------------------------------------------------------------------------


class QuaternionCanonicalizer:
    """Sequential per-episode quaternion sign fixer.

    ``q`` and ``-q`` are the same rotation, but the recorded stream flips sign
    occasionally (374 times across the three merged tasks -- 284 on the left arm,
    90 on the right), and a flip is a ``|dq| = 2`` discontinuity, roughly 30x the
    largest genuine step in this data (0.064). The rule is: within an episode,
    keep each state quaternion on the same side of the 4-sphere as its
    predecessor, then force the action quaternion of the same frame to agree
    with the state so the two can never end up in opposite hemispheres.

    State is carried across parquet file boundaries so an episode split over two
    files is treated as one continuous run. Processing is in global index order,
    which makes the result deterministic and exactly reproducible by the verify
    pass.
    """

    def __init__(self) -> None:
        self._episode: int | None = None
        self._previous: list[np.ndarray | None] = [None] * len(QUAT_SLICES)

    def reset(self) -> None:
        self._episode = None
        self._previous = [None] * len(QUAT_SLICES)

    def apply(
        self, state: np.ndarray, action: np.ndarray, episodes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        state = np.array(state, dtype=np.float64, copy=True)
        action = np.array(action, dtype=np.float64, copy=True)
        for row in range(len(state)):
            episode = int(episodes[row])
            if episode != self._episode:
                self._episode = episode
                self._previous = [None] * len(QUAT_SLICES)
            for arm, quat in enumerate(QUAT_SLICES):
                q = state[row, quat]
                previous = self._previous[arm]
                if previous is not None and float(np.dot(q, previous)) < 0.0:
                    q = -q
                    state[row, quat] = q
                self._previous[arm] = q
                # Tie the action to the state's hemisphere for this frame.
                if float(np.dot(action[row, quat], q)) < 0.0:
                    action[row, quat] = -action[row, quat]
        return state, action


def _check_quaternions(values: np.ndarray, label: str, atol: float = 1e-4) -> None:
    for arm, quat in enumerate(QUAT_SLICES):
        norms = np.linalg.norm(values[:, quat], axis=1)
        if not np.allclose(norms, 1.0, atol=atol, rtol=0):
            worst = float(np.abs(norms - 1.0).max())
            raise ValueError(f"{label} arm {arm} quaternion is not unit norm (max dev {worst:.2e})")


def build_canonical_columns(
    table: pa.Table, canonicalizer: QuaternionCanonicalizer
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonicalized ``(state, action)`` float32 arrays for one table."""
    state = _vector_column(table, SOURCE_STATE_KEY, STATE_DIM)
    action = _vector_column(table, SOURCE_ACTION_KEY, ACTION_DIM)
    _check_quaternions(state, SOURCE_STATE_KEY)
    _check_quaternions(action, SOURCE_ACTION_KEY)
    episodes = table.column("episode_index").combine_chunks().to_numpy(zero_copy_only=False)
    state, action = canonicalizer.apply(state, action, episodes)
    state32 = state.astype(np.float32)
    action32 = action.astype(np.float32)
    for label, values in ((STATE_KEY, state32), (ACTION_KEY, action32)):
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite value while constructing {label}")
    return state32, action32


def _fixed_list(values: np.ndarray, width: int) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float32()), width
    )


def to_canonical_table(
    table: pa.Table, canonicalizer: QuaternionCanonicalizer
) -> pa.Table:
    """Rewrite one source table into the canonical derived schema."""
    for key in (SOURCE_STATE_KEY, SOURCE_ACTION_KEY, *IDENTITY_KEYS):
        if key not in table.column_names:
            raise ValueError(f"source table is missing required column {key!r}")
    state, action = build_canonical_columns(table, canonicalizer)
    columns = {key: table.column(key).combine_chunks() for key in IDENTITY_KEYS}
    arrays = [_fixed_list(state, STATE_DIM), _fixed_list(action, ACTION_DIM)]
    names = [STATE_KEY, ACTION_KEY]
    for key, column in columns.items():
        arrays.append(column)
        names.append(key)
    return pa.Table.from_arrays(arrays, names=names)


DERIVED_COLUMNS: tuple[str, ...] = (STATE_KEY, ACTION_KEY, *IDENTITY_KEYS)


# --------------------------------------------------------------------------
# Source identity
# --------------------------------------------------------------------------


def source_content_digest(source_root: Path) -> str:
    """Content hash over ``meta/`` and every referenced data parquet.

    A locally recorded dataset has no Hugging Face revision marker to pin, and
    hashing only metadata is what let 33 truncated DROID parquet files pass
    every provenance check and die inside the dataloader.
    """
    digest = hashlib.sha256()
    meta = source_root / "meta"
    if not meta.is_dir():
        raise FileNotFoundError(meta)
    members = sorted(
        path for path in meta.rglob("*") if path.is_file() and not path.name.startswith(".")
    )
    members.extend(sorted((source_root / "data").glob("chunk-*/*.parquet")))
    if not members:
        raise ValueError(f"{source_root} contains no metadata or data files")
    for path in members:
        digest.update(str(path.relative_to(source_root)).encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _check_source_digest(source_root: Path, expected: str) -> None:
    actual = source_content_digest(source_root)
    if actual != expected:
        raise ValueError(
            f"source content digest is {actual}, expected {expected}; "
            "the source dataset changed since this build was configured"
        )


def _episode_metadata(source_root: Path) -> pa.Table:
    paths = sorted((source_root / "meta/episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"{source_root}/meta/episodes has no parquet files")
    columns = [
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "data/chunk_index",
        "data/file_index",
    ]
    return pa.concat_tables([pq.read_table(path, columns=columns) for path in paths])


def _source_parquets(source_root: Path) -> list[Path]:
    """Resolve data files from episode metadata, excluding stale local files."""
    info = json.loads((source_root / "meta/info.json").read_text(encoding="utf-8"))
    template = info.get("data_path")
    if not isinstance(template, str):
        raise ValueError("meta/info.json has no data_path template")
    if int(info.get("fps", -1)) != EXPECTED_FPS:
        raise ValueError(f"expected fps={EXPECTED_FPS}, got {info.get('fps')!r}")

    episodes = _episode_metadata(source_root)
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError(
            f"episode metadata has {len(episodes)} rows, expected {info['total_episodes']}"
        )
    references = sorted(
        set(
            zip(
                map(int, episodes.column("data/chunk_index").to_pylist()),
                map(int, episodes.column("data/file_index").to_pylist()),
                strict=True,
            )
        )
    )
    if not references:
        raise ValueError("episode metadata references no data parquet files")

    files = [
        source_root / template.format(chunk_index=chunk_index, file_index=file_index)
        for chunk_index, file_index in references
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"episode metadata references missing data files: {missing[:3]}")

    discovered = set((source_root / "data").glob("chunk-*/*.parquet"))
    ignored = sorted(discovered.difference(files))
    if ignored:
        print(
            f"[prepare] ignoring {len(ignored)} unreferenced local parquet files; "
            "episode metadata is authoritative",
            flush=True,
        )
    frame_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    if frame_rows != int(info["total_frames"]):
        raise ValueError(
            f"referenced data files have {frame_rows} rows, expected {info['total_frames']}"
        )
    return files


def plan_source_files(source_root: Path, max_episodes: int | None = None) -> list[FilePlan]:
    """Select complete leading rows without renumbering any identifier.

    These recordings carry no ``is_last`` column, so episode boundaries come
    from ``dataset_from_index``/``dataset_to_index``, which is authoritative.
    """
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")

    files = _source_parquets(source_root)
    file_rows = [pq.ParquetFile(path).metadata.num_rows for path in files]
    if max_episodes is None:
        return [
            FilePlan(str(path.relative_to(source_root)), rows)
            for path, rows in zip(files, file_rows, strict=True)
        ]

    episodes = _episode_metadata(source_root)
    episode_ids = np.asarray(episodes.column("episode_index").to_pylist(), dtype=np.int64)
    to_index = np.asarray(episodes.column("dataset_to_index").to_pylist(), dtype=np.int64)
    order = np.argsort(episode_ids, kind="stable")
    if max_episodes > len(order):
        raise ValueError(f"source contains fewer than {max_episodes} complete episodes")
    keep_rows = int(to_index[order[max_episodes - 1]])

    plans: list[FilePlan] = []
    consumed = 0
    for path, rows in zip(files, file_rows, strict=True):
        if consumed >= keep_rows:
            break
        take = min(rows, keep_rows - consumed)
        plans.append(FilePlan(str(path.relative_to(source_root)), take))
        consumed += take
    if consumed != keep_rows:
        raise ValueError(f"data files hold {consumed} rows, expected {keep_rows}")
    return plans


def _selected_episode_max(source_root: Path, plans: list[FilePlan]) -> int:
    final = plans[-1]
    episodes = pq.read_table(
        source_root / final.relative_path, columns=["episode_index"]
    ).column("episode_index")
    if final.rows <= 0:
        raise ValueError("selected data prefix is empty")
    return int(episodes[final.rows - 1].as_py())


def _copy_metadata(
    source_root: Path,
    output_root: Path,
    selected_episode_max: int,
    partial: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Copy v3 metadata, filtering episode rows only for a smoke prefix."""
    source_meta = source_root / "meta"
    output_meta = output_root / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source in sorted(source_meta.iterdir()):
        if source.is_file():
            destination = output_meta / source.name
            shutil.copy2(source, destination)
            copied.append(str(destination.relative_to(output_root)))

    episode_records: list[dict[str, Any]] = []
    source_episodes = source_meta / "episodes"
    if not source_episodes.is_dir():
        raise FileNotFoundError(source_episodes)
    for source in sorted(source_episodes.glob("chunk-*/*.parquet")):
        relative = source.relative_to(source_root)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        output_table = None
        if partial:
            source_table = pq.read_table(source)
            episode_ids = source_table.column("episode_index").to_numpy(zero_copy_only=False)
            keep = int(np.searchsorted(episode_ids, selected_episode_max, side="right"))
            if keep == 0:
                break
            output_table = source_table.slice(0, keep)
            pq.write_table(output_table, destination, compression="zstd")
            source_rows = len(source_table)
            output_rows = len(output_table)
        else:
            shutil.copy2(source, destination)
            source_rows = pq.ParquetFile(source).metadata.num_rows
            output_rows = source_rows
        copied.append(str(relative))
        episode_records.append(
            {
                "path": str(relative),
                "source_rows": source_rows,
                "output_rows": output_rows,
                "source_sha256": _sha256(source),
                "output_sha256": _sha256(destination),
            }
        )
        if (
            partial
            and output_table is not None
            and int(output_table.column("episode_index")[-1].as_py()) >= selected_episode_max
        ):
            break
    if not episode_records:
        raise ValueError("no episode metadata rows were copied")
    last_episode = int(
        pq.read_table(output_root / episode_records[-1]["path"], columns=["episode_index"])
        .column("episode_index")[-1]
        .as_py()
    )
    if partial and last_episode != selected_episode_max:
        raise ValueError(
            f"episode metadata ended at {last_episode}, expected {selected_episode_max}"
        )
    return copied, episode_records


def _create_video_symlink(
    source_root: Path, staging_root: Path, final_root: Path, mode: str
) -> str:
    source_videos = source_root / "videos"
    if not source_videos.is_dir():
        raise FileNotFoundError(source_videos)
    link = staging_root / "videos"
    if mode == "absolute":
        target = str(source_videos.resolve())
    elif mode == "relative":
        target = os.path.relpath(source_videos.resolve(), start=final_root.resolve())
    else:
        raise ValueError(f"unknown video symlink mode: {mode}")
    link.symlink_to(target, target_is_directory=True)
    return target


def _convert_files(
    source_root: Path, output_root: Path, plans: list[FilePlan]
) -> list[dict[str, Any]]:
    """Rewrite each planned parquet into the canonical schema.

    A single canonicalizer instance spans all files so an episode split across a
    file boundary keeps one continuous quaternion sign convention.
    """
    canonicalizer = QuaternionCanonicalizer()
    records: list[dict[str, Any]] = []
    for file_number, plan in enumerate(plans, start=1):
        source = source_root / plan.relative_path
        destination = output_root / plan.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(source)
        if plan.rows > len(table):
            raise ValueError(f"planned rows exceed source rows for {source}")
        table = table.slice(0, plan.rows)
        converted = to_canonical_table(table, canonicalizer)
        pq.write_table(converted, destination, compression="zstd")
        records.append(
            {
                "path": plan.relative_path,
                "source_rows": pq.ParquetFile(source).metadata.num_rows,
                "output_rows": plan.rows,
                "source_sha256": _sha256(source),
                "output_sha256": _sha256(destination),
            }
        )
        if file_number == 1 or file_number % 25 == 0 or file_number == len(plans):
            print(f"[prepare] converted parquet files {file_number}/{len(plans)}", flush=True)
    return records


def _episode_tables(root: Path, plans: list[FilePlan], columns: list[str]) -> Iterator[pa.Table]:
    carry: pa.Table | None = None
    for plan in plans:
        table = pq.read_table(root / plan.relative_path, columns=columns).slice(0, plan.rows)
        if not len(table):
            continue
        episode_ids = table.column("episode_index").to_numpy(zero_copy_only=False)
        boundaries = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
        starts = np.r_[0, boundaries]
        stops = np.r_[boundaries, len(table)]
        groups = [table.slice(int(start), int(stop - start)) for start, stop in zip(starts, stops)]
        if carry is not None:
            carry_id = int(carry.column("episode_index")[0].as_py())
            first_id = int(groups[0].column("episode_index")[0].as_py())
            if carry_id == first_id:
                groups[0] = pa.concat_tables([carry, groups[0]])
            else:
                yield carry
        yield from groups[:-1]
        carry = groups[-1]
    if carry is not None:
        yield carry


def _episode_arrays(table: pa.Table) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        key: table.column(key).combine_chunks().to_numpy(zero_copy_only=False)
        for key in IDENTITY_KEYS
    }
    arrays[STATE_KEY] = _vector_column(table, STATE_KEY, STATE_DIM).astype(np.float32, copy=False)
    arrays[ACTION_KEY] = _vector_column(table, ACTION_KEY, ACTION_DIM).astype(
        np.float32, copy=False
    )
    return arrays


def _valid_task_indices(root: Path) -> set[int]:
    """Return task IDs whose canonical LeRobot task string is non-empty."""
    path = root / "meta/tasks.parquet"
    table = pq.read_table(path)
    if "task_index" not in table.column_names:
        raise ValueError(f"{path} is missing task_index")
    text_key = "task" if "task" in table.column_names else "__index_level_0__"
    if text_key not in table.column_names:
        raise ValueError(f"{path} has no canonical task-text column")
    task_indices = table.column("task_index").to_pylist()
    task_texts = table.column(text_key).to_pylist()
    if len(task_indices) != len(set(task_indices)):
        raise ValueError(f"{path} contains duplicate task_index values")
    return {
        int(task_index)
        for task_index, task in zip(task_indices, task_texts, strict=True)
        if task is not None and str(task).strip()
    }


def _true_segments(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    differences = np.diff(padded.astype(np.int8))
    return np.flatnonzero(differences == 1), np.flatnonzero(differences == -1)


def non_idle_anchor_mask(
    actions: np.ndarray,
    config: NonIdleConfig,
    horizon: int = HORIZON,
) -> np.ndarray:
    """Idle segmentation over the consecutive-command delta of the action."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"expected action shape (N, {ACTION_DIM}), got {actions.shape}")
    anchor_count = max(0, len(actions) - horizon)
    anchors = np.zeros(anchor_count, dtype=bool)
    if anchor_count == 0:
        return anchors

    is_idle = np.concatenate(
        ([False], np.all(np.abs(actions[1:] - actions[:-1]) < config.action_delta, axis=1))
    )
    idle_starts, idle_ends = _true_segments(is_idle)
    keep = np.ones(len(actions), dtype=bool)
    for start, end in zip(idle_starts, idle_ends, strict=True):
        if end - start >= config.min_idle_len:
            keep[start:end] = False

    keep_starts, keep_ends = _true_segments(keep)
    for start, end in zip(keep_starts, keep_ends, strict=True):
        if end - start < config.min_non_idle_len:
            continue
        anchor_end = min(end - config.trim_last_steps, anchor_count)
        if anchor_end > start:
            anchors[start:anchor_end] = True
    return anchors


def anchor_mask(
    arrays: dict[str, np.ndarray],
    non_idle: NonIdleConfig,
    horizon: int = HORIZON,
    valid_task_indices: set[int] | None = None,
    excluded_episodes: frozenset[int] = frozenset(),
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid-anchor mask and ``(N, horizon, 16)`` action windows."""
    episode = arrays["episode_index"]
    frame = arrays["frame_index"]
    index = arrays["index"]
    timestamp = arrays["timestamp"]
    state = arrays[STATE_KEY]
    actions = arrays[ACTION_KEY]
    if not np.all(episode == episode[0]):
        raise ValueError("episode scanner emitted mixed episode IDs")
    if not np.all(np.diff(frame) == 1) or not np.all(np.diff(index) == 1):
        raise ValueError(f"non-contiguous indices in episode {int(episode[0])}")
    if not np.all(np.diff(timestamp) > 0):
        raise ValueError(f"non-increasing timestamps in episode {int(episode[0])}")
    for key, values in ((STATE_KEY, state), (ACTION_KEY, actions)):
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite {key} in episode {int(episode[0])}")
    task_indices = arrays["task_index"]
    if not np.all(task_indices == task_indices[0]):
        raise ValueError(f"task_index changes within episode {int(episode[0])}")

    length = len(index)
    count = max(0, length - horizon)
    if count == 0:
        return np.zeros(0, dtype=bool), np.empty((0, horizon, ACTION_DIM), dtype=np.float32)

    included = int(episode[0]) not in excluded_episodes
    eligible = np.full(count, included, dtype=bool)
    if valid_task_indices is None:
        raise ValueError("valid_task_indices is required")
    eligible &= np.isin(task_indices[:count], list(valid_task_indices))

    windows = sliding_window_view(actions, horizon, axis=0)[:count].transpose(0, 2, 1)
    moving = non_idle_anchor_mask(actions, non_idle, horizon)
    return eligible & moving, windows


def _anchor_columns(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "index": arrays["index"][: len(mask)][mask].astype(np.int64, copy=False),
        "episode_index": arrays["episode_index"][: len(mask)][mask].astype(np.int64, copy=False),
        "frame_index": arrays["frame_index"][: len(mask)][mask].astype(np.int64, copy=False),
        "task_index": arrays["task_index"][: len(mask)][mask].astype(np.int64, copy=False),
        "timestamp": arrays["timestamp"][: len(mask)][mask].astype(np.float32, copy=False),
    }


def _scan_columns() -> list[str]:
    return list(dict.fromkeys([*IDENTITY_KEYS, STATE_KEY, ACTION_KEY]))


def _write_anchor_artifacts(
    root: Path,
    plans: list[FilePlan],
    non_idle: NonIdleConfig,
    smoke_count: int,
    excluded_episodes: frozenset[int],
) -> tuple[int, list[int], int]:
    manifest_path = root / "valid_anchor_indices.parquet"
    writer = pq.ParquetWriter(manifest_path, ANCHOR_SCHEMA, compression="zstd")
    valid_task_indices = _valid_task_indices(root)
    if not valid_task_indices:
        raise ValueError("canonical task table contains no non-empty language")
    included_episodes: list[int] = []
    smoke_rows: list[dict[str, Any]] = []
    anchor_count = 0
    included_frames = 0
    try:
        for episode_number, episode_table in enumerate(
            _episode_tables(root, plans, _scan_columns()), start=1
        ):
            arrays = _episode_arrays(episode_table)
            episode_id = int(arrays["episode_index"][0])
            if episode_id not in excluded_episodes:
                included_episodes.append(episode_id)
                included_frames += len(arrays["index"])
            mask, _ = anchor_mask(
                arrays,
                non_idle,
                valid_task_indices=valid_task_indices,
                excluded_episodes=excluded_episodes,
            )
            columns = _anchor_columns(arrays, mask)
            count = len(columns["index"])
            if count:
                batch = pa.Table.from_pydict(columns, schema=ANCHOR_SCHEMA)
                writer.write_table(batch)
                if len(smoke_rows) < smoke_count:
                    take = min(smoke_count - len(smoke_rows), count)
                    smoke_rows.extend(batch.slice(0, take).to_pylist())
                anchor_count += count
            if episode_number % 25 == 0:
                print(
                    f"[prepare] scanned {episode_number} episodes; valid anchors={anchor_count}",
                    flush=True,
                )
    finally:
        writer.close()
    _json_dump(
        root / "included_episodes.json",
        {"count": len(included_episodes), "episode_indices": included_episodes},
    )
    _json_dump(
        root / "smoke_anchor_indices.json",
        {
            "selection": "first_valid_anchors_in_global_index_order",
            "requested": smoke_count,
            "count": len(smoke_rows),
            "anchors": smoke_rows,
        },
    )
    return anchor_count, included_episodes, included_frames


def _merge_moments(
    count: int, mean: np.ndarray, m2: np.ndarray, chunk: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray]:
    chunk64 = np.asarray(chunk, dtype=np.float64)
    chunk_count = len(chunk64)
    if not chunk_count:
        return count, mean, m2
    chunk_mean = chunk64.mean(axis=0)
    chunk_m2 = np.square(chunk64 - chunk_mean).sum(axis=0)
    if count == 0:
        return chunk_count, chunk_mean, chunk_m2
    total = count + chunk_count
    delta = chunk_mean - mean
    merged_mean = mean + delta * (chunk_count / total)
    merged_m2 = m2 + chunk_m2 + delta * delta * (count * chunk_count / total)
    return total, merged_mean, merged_m2


def _quantiles_from_sorted(sorted_values: np.memmap) -> list[float]:
    size = len(sorted_values)
    if size == 0:
        raise ValueError("cannot compute quantiles of an empty feature")
    values: list[float] = []
    for quantile in QUANTILES:
        position = (size - 1) * quantile
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        fraction = position - lower
        value = float(sorted_values[lower]) * (1 - fraction) + float(
            sorted_values[upper]
        ) * fraction
        values.append(value)
    return values


def _memmap_stats(
    data_path: Path,
    shape: tuple[int, int],
    scratch_dir: Path,
    chunk_rows: int = 262_144,
) -> dict[str, list[float] | list[int]]:
    data = np.memmap(data_path, dtype=np.float32, mode="r", shape=shape)
    minimum = np.full(shape[1], np.inf, dtype=np.float64)
    maximum = np.full(shape[1], -np.inf, dtype=np.float64)
    count = 0
    mean = np.zeros(shape[1], dtype=np.float64)
    m2 = np.zeros(shape[1], dtype=np.float64)
    for start in range(0, shape[0], chunk_rows):
        chunk = np.asarray(data[start : start + chunk_rows])
        if not np.isfinite(chunk).all():
            raise ValueError(f"non-finite values in stats memmap {data_path}")
        minimum = np.minimum(minimum, chunk.min(axis=0))
        maximum = np.maximum(maximum, chunk.max(axis=0))
        count, mean, m2 = _merge_moments(count, mean, m2, chunk)

    quantile_rows: list[list[float]] = [[] for _ in QUANTILES]
    scratch_path = scratch_dir / f"{data_path.stem}.sorted.f32"
    for dimension in range(shape[1]):
        scratch = np.memmap(scratch_path, dtype=np.float32, mode="w+", shape=(shape[0],))
        for start in range(0, shape[0], chunk_rows):
            stop = min(start + chunk_rows, shape[0])
            scratch[start:stop] = data[start:stop, dimension]
        scratch.flush()
        scratch.sort()
        for row, value in zip(quantile_rows, _quantiles_from_sorted(scratch), strict=True):
            row.append(value)
        del scratch
    scratch_path.unlink(missing_ok=True)
    del data
    result: dict[str, list[float] | list[int]] = {
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(m2 / count).tolist(),
        "count": [count],
    }
    for quantile, row in zip(QUANTILES, quantile_rows, strict=True):
        result[f"q{int(quantile * 100):02d}"] = row
    return result


def _compute_feature_stats(
    root: Path,
    plans: list[FilePlan],
    included_frames: int,
    work_dir: Path,
    excluded_episodes: frozenset[int],
) -> dict[str, Any]:
    """Recompute state/action statistics over every frame of included episodes.

    This is the LIBERO ``fix_stats.sh`` arrangement rather than DROID's
    anchor-only one, because the goal target *is* ``observation.state`` at
    ``t + H``: current states and goal states must share one set of quantiles,
    so the statistic has to cover every frame either role can reach.

    It also exists because LeRobot v3.0 writes ``meta/stats.json`` by
    aggregating per-episode statistics, which is not the true global quantile.
    The three merged sources shipped exactly that defect: ``dustpan_filtered``
    would clip 24.2% of one state dimension and ``transfer_filtered`` 16.0%.
    ``merge_datasets.py`` already recomputes an exact global statistic, so this
    pass re-derives it a second time over the included frames only, which is
    the scope the goal target actually needs.
    """
    if included_frames <= 0:
        raise ValueError("no included frames; refusing to produce empty statistics")
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        STATE_KEY: work_dir / "frames_state.f32",
        ACTION_KEY: work_dir / "frames_action.f32",
    }
    shapes = {
        STATE_KEY: (included_frames, STATE_DIM),
        ACTION_KEY: (included_frames, ACTION_DIM),
    }
    maps = {
        key: np.memmap(paths[key], dtype=np.float32, mode="w+", shape=shapes[key])
        for key in paths
    }
    offset = 0
    for episode_table in _episode_tables(root, plans, _scan_columns()):
        arrays = _episode_arrays(episode_table)
        if int(arrays["episode_index"][0]) in excluded_episodes:
            continue
        rows = len(arrays["index"])
        stop = offset + rows
        maps[STATE_KEY][offset:stop] = arrays[STATE_KEY]
        maps[ACTION_KEY][offset:stop] = arrays[ACTION_KEY]
        offset = stop
    if offset != included_frames:
        raise ValueError(f"frame scan changed: expected {included_frames}, got {offset}")
    for mapping in maps.values():
        mapping.flush()
    maps.clear()

    features = {key: _memmap_stats(paths[key], shapes[key], work_dir) for key in paths}
    for path in paths.values():
        path.unlink(missing_ok=True)
    return {
        "included_frames": included_frames,
        "horizon": HORIZON,
        "scope": "all frames of included episodes",
        "features": features,
    }


def _update_metadata(
    root: Path,
    stats: dict[str, Any],
    partial: bool,
    total_frames: int,
    total_episodes: int,
    anchor_count: int,
) -> None:
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.setdefault("features", {})
    for key in (*DROPPED_SOURCE_KEYS, SOURCE_STATE_KEY, SOURCE_ACTION_KEY):
        features.pop(key, None)
    # The per-dimension names drive the MolmoAct2 gripper normalization mask
    # ("gripper" not in name.lower()), so both gripper dims must stay named.
    for key, dim in ((STATE_KEY, STATE_DIM), (ACTION_KEY, ACTION_DIM)):
        features[key] = {
            "dtype": "float32",
            "shape": [dim],
            "names": list(FEATURE_NAMES),
        }
    info["goal_pose_prior"] = {
        "horizon": HORIZON,
        "partial_build": partial,
        "anchor_manifest": "valid_anchor_indices.parquet",
        "target_feature": STATE_KEY,
        "anchor_count": anchor_count,
    }
    if partial:
        info["total_frames"] = total_frames
        info["total_episodes"] = total_episodes
        info["splits"] = {"train": f"0:{total_episodes}"}
    _json_dump(info_path, info)

    stats_path = root / "meta/stats.json"
    source_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for key in (*DROPPED_SOURCE_KEYS, SOURCE_STATE_KEY, SOURCE_ACTION_KEY):
        source_stats.pop(key, None)
    source_stats.update(stats["features"])
    _json_dump(stats_path, source_stats)
    _json_dump(root / "anchor_stats.json", stats)


def _artifact_hashes(root: Path, files: list[dict[str, Any]]) -> dict[str, str]:
    """Hash every published artifact, including the derived data parquets."""
    names = [
        "included_episodes.json",
        "valid_anchor_indices.parquet",
        "smoke_anchor_indices.json",
        "anchor_stats.json",
    ]
    names.extend(
        str(path.relative_to(root)) for path in sorted((root / "meta").iterdir()) if path.is_file()
    )
    names.extend(record["path"] for record in files)
    return {name: _sha256(root / name) for name in names}


class _ManifestCursor:
    def __init__(self, path: Path) -> None:
        self._iterator = pq.ParquetFile(path).iter_batches(batch_size=65_536)
        self._batch: pa.RecordBatch | None = None
        self._offset = 0
        self.rows = 0

    def _ensure_batch(self) -> bool:
        while self._batch is None or self._offset == len(self._batch):
            try:
                self._batch = next(self._iterator)
                self._offset = 0
            except StopIteration:
                self._batch = None
                return False
        return True

    def consume(self, expected: dict[str, np.ndarray]) -> None:
        expected_offset = 0
        count = len(expected["index"])
        while expected_offset < count:
            if not self._ensure_batch():
                raise ValueError("anchor manifest ended early")
            assert self._batch is not None
            take = min(count - expected_offset, len(self._batch) - self._offset)
            for key in ANCHOR_SCHEMA.names:
                actual = (
                    self._batch.column(key)
                    .slice(self._offset, take)
                    .to_numpy(zero_copy_only=False)
                )
                wanted = expected[key][expected_offset : expected_offset + take]
                if not np.array_equal(actual, wanted):
                    raise ValueError(f"anchor manifest mismatch in {key}")
            self._offset += take
            expected_offset += take
            self.rows += take

    def finish(self) -> None:
        if self._ensure_batch():
            raise ValueError("anchor manifest contains extra rows")


def _verify_derived_columns(
    source_root: Path,
    output_root: Path,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute the canonical columns from source and require exact equality."""
    previous: tuple[int, int, int, float] | None = None
    finished_episodes: set[int] = set()
    frame_period = 1.0 / EXPECTED_FPS
    canonicalizer = QuaternionCanonicalizer()
    flips_fixed = 0
    for record in files:
        relative = record["path"]
        source_path = source_root / relative
        output_path = output_root / relative
        if _sha256(source_path) != record["source_sha256"]:
            raise ValueError(f"source hash changed: {relative}")
        if _sha256(output_path) != record["output_sha256"]:
            raise ValueError(f"derived hash changed: {relative}")
        if pq.ParquetFile(source_path).metadata.num_rows != int(record["source_rows"]):
            raise ValueError(f"source row-count changed: {relative}")
        if pq.ParquetFile(output_path).metadata.num_rows != int(record["output_rows"]):
            raise ValueError(f"derived row-count changed: {relative}")
        source = pq.read_table(source_path).slice(0, int(record["output_rows"]))
        derived = pq.read_table(output_path)
        if len(source) != len(derived):
            raise ValueError(f"row-count mismatch: {relative}")
        if tuple(derived.column_names) != DERIVED_COLUMNS:
            raise ValueError(
                f"derived schema is {derived.column_names}, expected {list(DERIVED_COLUMNS)}"
            )

        raw_state = _vector_column(source, SOURCE_STATE_KEY, STATE_DIM)
        raw_action = _vector_column(source, SOURCE_ACTION_KEY, ACTION_DIM)
        expected_state, expected_action = build_canonical_columns(source, canonicalizer)
        actual_state = _vector_column(derived, STATE_KEY, STATE_DIM)
        actual_action = _vector_column(derived, ACTION_KEY, ACTION_DIM)
        for label, actual, expected in (
            (STATE_KEY, actual_state, expected_state),
            (ACTION_KEY, actual_action, expected_action),
        ):
            if actual.dtype != np.float32:
                raise ValueError(f"{label} is not float32: {relative}")
            if not np.array_equal(actual, expected):
                raise ValueError(f"{label} does not reproduce from source: {relative}")
            if not np.isfinite(actual).all():
                raise ValueError(f"non-finite {label}: {relative}")

        # Canonicalization must only ever flip signs: the rotation itself, the
        # translation and the gripper must be untouched.
        for arm, quat in enumerate(QUAT_SLICES):
            if not np.allclose(
                np.abs(actual_state[:, quat]), np.abs(raw_state[:, quat]), atol=1e-6, rtol=0
            ):
                raise ValueError(f"state quaternion magnitude changed on arm {arm}: {relative}")
            if not np.allclose(
                np.abs(actual_action[:, quat]), np.abs(raw_action[:, quat]), atol=1e-6, rtol=0
            ):
                raise ValueError(f"action quaternion magnitude changed on arm {arm}: {relative}")
            flips_fixed += int((np.sign(actual_state[:, quat.start]) != np.sign(
                raw_state[:, quat.start]
            )).sum())
        keep = [i for i in range(STATE_DIM) if not any(q.start <= i < q.stop for q in QUAT_SLICES)]
        if not np.allclose(actual_state[:, keep], raw_state[:, keep], atol=1e-6, rtol=0):
            raise ValueError(f"non-rotation state dims changed: {relative}")
        if not np.allclose(actual_action[:, keep], raw_action[:, keep], atol=1e-6, rtol=0):
            raise ValueError(f"non-rotation action dims changed: {relative}")
        _check_quaternions(actual_state, STATE_KEY)
        _check_quaternions(actual_action, ACTION_KEY)

        # Identity columns must survive untouched.
        for key in IDENTITY_KEYS:
            if not derived.column(key).equals(source.column(key)):
                raise ValueError(f"identity column {key} changed: {relative}")

        identities = {
            key: derived.column(key).to_numpy(zero_copy_only=False) for key in IDENTITY_KEYS
        }
        for row in zip(
            identities["episode_index"],
            identities["frame_index"],
            identities["index"],
            identities["timestamp"],
            strict=True,
        ):
            current = (int(row[0]), int(row[1]), int(row[2]), float(row[3]))
            if previous is not None:
                if current[0] == previous[0]:
                    if current[1] != previous[1] + 1 or current[2] != previous[2] + 1:
                        raise ValueError(f"non-contiguous episode {current[0]}")
                    if not np.isclose(current[3] - previous[3], frame_period, atol=2e-5, rtol=0):
                        raise ValueError(f"timestamp discontinuity in episode {current[0]}")
                else:
                    finished_episodes.add(previous[0])
                    if current[0] in finished_episodes:
                        raise ValueError(f"episode {current[0]} is not stored contiguously")
            previous = current
    return {"quaternion_signs_flipped": flips_fixed}


def _verify_no_residual_flips(root: Path, plans: list[FilePlan]) -> int:
    """Assert the published quaternions are continuous within every episode."""
    worst = 0.0
    for episode_table in _episode_tables(root, plans, _scan_columns()):
        arrays = _episode_arrays(episode_table)
        for key in (STATE_KEY, ACTION_KEY):
            values = arrays[key]
            for quat in QUAT_SLICES:
                q = values[:, quat].astype(np.float64)
                if len(q) < 2:
                    continue
                dots = np.sum(q[1:] * q[:-1], axis=1)
                if np.any(dots < 0):
                    raise ValueError(
                        f"{key} still contains a quaternion sign flip in episode "
                        f"{int(arrays['episode_index'][0])}"
                    )
                worst = max(worst, float(np.linalg.norm(q[1:] - q[:-1], axis=1).max()))
    return worst


def _verify_episode_metadata(
    source_root: Path,
    output_root: Path,
    records: list[dict[str, Any]],
) -> None:
    previous_episode: int | None = None
    for record in records:
        relative = record["path"]
        source_path = source_root / relative
        output_path = output_root / relative
        if _sha256(source_path) != record["source_sha256"]:
            raise ValueError(f"source episode metadata hash changed: {relative}")
        if _sha256(output_path) != record["output_sha256"]:
            raise ValueError(f"derived episode metadata hash changed: {relative}")
        if pq.ParquetFile(source_path).metadata.num_rows != int(record["source_rows"]):
            raise ValueError(f"source episode metadata row count changed: {relative}")
        if pq.ParquetFile(output_path).metadata.num_rows != int(record["output_rows"]):
            raise ValueError(f"derived episode metadata row count changed: {relative}")
        if record["source_sha256"] != record["output_sha256"] or int(
            record["source_rows"]
        ) != int(record["output_rows"]):
            source = pq.read_table(source_path).slice(0, int(record["output_rows"]))
            output = pq.read_table(output_path)
            if not output.equals(source):
                raise ValueError(f"episode metadata pairing mismatch: {relative}")
        episodes = (
            pq.read_table(output_path, columns=["episode_index"])
            .column("episode_index")
            .to_numpy(zero_copy_only=False)
        )
        if len(episodes) and previous_episode is not None:
            if int(episodes[0]) != previous_episode + 1:
                raise ValueError("episode metadata is not globally contiguous")
        if len(episodes):
            if not np.all(np.diff(episodes) == 1):
                raise ValueError(f"episode metadata is not contiguous: {relative}")
            previous_episode = int(episodes[-1])


def _assert_stats_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    if actual.keys() != expected.keys():
        raise ValueError("stats top-level keys differ")
    for key in ("included_frames", "horizon", "scope"):
        if actual[key] != expected[key]:
            raise ValueError(f"stats {key} mismatch")
    if actual["features"].keys() != expected["features"].keys():
        raise ValueError("stats feature keys differ")
    for feature in expected["features"]:
        for statistic, wanted in expected["features"][feature].items():
            got = actual["features"][feature].get(statistic)
            if statistic == "count":
                if got != wanted:
                    raise ValueError(f"{feature} {statistic} mismatch")
            elif not np.allclose(got, wanted, atol=1e-7, rtol=1e-7):
                raise ValueError(f"{feature} {statistic} mismatch")


def _load_build_config(config_dict: dict[str, Any]) -> BuildConfig:
    return BuildConfig(
        source_root=config_dict["source_root"],
        output_root=config_dict["output_root"],
        source_digest=config_dict["source_digest"],
        max_episodes=config_dict["max_episodes"],
        exclude_episodes=tuple(int(item) for item in config_dict["exclude_episodes"]),
        video_symlink=config_dict["video_symlink"],
        smoke_anchors=config_dict["smoke_anchors"],
        horizon=config_dict["horizon"],
        non_idle=NonIdleConfig(**config_dict["non_idle"]),
    )


def verify_dataset(
    root: Path,
    source_root: Path | None = None,
    require_success: bool = True,
) -> dict[str, Any]:
    """Fail-closed verification of columns, anchors, stats, and hashes."""
    provenance_path = root / "goal_pose_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("recipe_version") != RECIPE_VERSION:
        raise ValueError(f"unsupported recipe version: {provenance.get('recipe_version')!r}")
    config_dict = provenance["config"]
    if provenance.get("config_fingerprint") != _config_dict_fingerprint(config_dict):
        raise ValueError("provenance config fingerprint mismatch")
    source = source_root or Path(config_dict["source_root"])
    _check_source_digest(source, config_dict["source_digest"])
    if config_dict["horizon"] != HORIZON:
        raise ValueError(f"unsupported horizon in provenance: {config_dict['horizon']}")
    loaded_config = _load_build_config(config_dict)
    _validate_config(loaded_config)
    non_idle = loaded_config.non_idle
    excluded = frozenset(loaded_config.exclude_episodes)
    plans = [FilePlan(item["path"], int(item["output_rows"])) for item in provenance["files"]]

    if require_success:
        success_path = root / "_SUCCESS.json"
        if not success_path.is_file():
            raise FileNotFoundError("atomic completion marker is missing")
        success = json.loads(success_path.read_text(encoding="utf-8"))
        if success["provenance_sha256"] != _sha256(provenance_path):
            raise ValueError("completion marker provenance hash mismatch")
        if success["config_fingerprint"] != provenance["config_fingerprint"]:
            raise ValueError("completion marker config mismatch")
    for relative, digest in provenance["artifact_sha256"].items():
        if _sha256(root / relative) != digest:
            raise ValueError(f"artifact hash changed: {relative}")

    _verify_derived_columns(source, root, provenance["files"])
    worst_step = _verify_no_residual_flips(root, plans)
    _verify_episode_metadata(source, root, provenance["episode_metadata_files"])

    cursor = _ManifestCursor(root / "valid_anchor_indices.parquet")
    valid_task_indices = _valid_task_indices(root)
    included: list[int] = []
    included_frames = 0
    expected_anchor_count = 0
    smoke_json = json.loads((root / "smoke_anchor_indices.json").read_text(encoding="utf-8"))
    smoke_requested = int(smoke_json["requested"])
    expected_smoke: list[dict[str, Any]] = []
    for episode_table in _episode_tables(root, plans, _scan_columns()):
        arrays = _episode_arrays(episode_table)
        if int(arrays["episode_index"][0]) not in excluded:
            included.append(int(arrays["episode_index"][0]))
            included_frames += len(arrays["index"])
        mask, _ = anchor_mask(
            arrays,
            non_idle,
            valid_task_indices=valid_task_indices,
            excluded_episodes=excluded,
        )
        expected = _anchor_columns(arrays, mask)
        cursor.consume(expected)
        if len(expected_smoke) < smoke_requested and len(expected["index"]):
            take = min(smoke_requested - len(expected_smoke), len(expected["index"]))
            expected_smoke.extend(
                pa.Table.from_pydict(expected, schema=ANCHOR_SCHEMA).slice(0, take).to_pylist()
            )
        expected_anchor_count += int(mask.sum())
    cursor.finish()
    if expected_anchor_count != provenance["anchor_count"]:
        raise ValueError("anchor count does not match provenance")

    included_json = json.loads((root / "included_episodes.json").read_text(encoding="utf-8"))
    if included_json != {"count": len(included), "episode_indices": included}:
        raise ValueError("included episode manifest mismatch")
    expected_smoke_json = {
        "selection": "first_valid_anchors_in_global_index_order",
        "requested": smoke_requested,
        "count": len(expected_smoke),
        "anchors": expected_smoke,
    }
    if smoke_json != expected_smoke_json:
        raise ValueError("smoke anchor manifest mismatch")

    with tempfile.TemporaryDirectory(prefix="yam-goal-prior-verify-") as temporary:
        recomputed_stats = _compute_feature_stats(
            root, plans, included_frames, Path(temporary), excluded
        )
    saved_stats = json.loads((root / "anchor_stats.json").read_text(encoding="utf-8"))
    _assert_stats_equal(saved_stats, recomputed_stats)
    metadata_stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
    for feature, expected_feature in recomputed_stats["features"].items():
        if feature not in metadata_stats:
            raise ValueError(f"meta/stats.json is missing {feature}")
        envelope = {
            "included_frames": included_frames,
            "horizon": HORIZON,
            "scope": recomputed_stats["scope"],
        }
        _assert_stats_equal(
            {**envelope, "features": {feature: metadata_stats[feature]}},
            {**envelope, "features": {feature: expected_feature}},
        )

    videos = root / "videos"
    if not videos.is_symlink() or videos.resolve() != (source / "videos").resolve():
        raise ValueError("videos is not a symlink to the source videos directory")
    return {
        "files": len(plans),
        "anchors": expected_anchor_count,
        "included_episodes": len(included),
        "included_frames": included_frames,
        "max_quaternion_step": worst_step,
    }


def prepare_dataset(config: BuildConfig) -> dict[str, Any]:
    _validate_config(config)
    source_root = Path(config.source_root).resolve()
    output_root = Path(config.output_root).resolve()
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("output root must not be the source root or a child of it")
    _check_source_digest(source_root, config.source_digest)
    fingerprint = _config_fingerprint(config)
    excluded = frozenset(config.exclude_episodes)

    if output_root.exists():
        provenance_path = output_root / "goal_pose_provenance.json"
        success_path = output_root / "_SUCCESS.json"
        if provenance_path.is_file() and success_path.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance.get("config_fingerprint") != fingerprint:
                raise FileExistsError("completed output exists with a different configuration")
            success = json.loads(success_path.read_text(encoding="utf-8"))
            if success.get("config_fingerprint") != fingerprint:
                raise ValueError("completion marker config mismatch")
            if success.get("provenance_sha256") != _sha256(provenance_path):
                raise ValueError("completion marker provenance hash mismatch")
            return {
                "files": len(provenance["files"]),
                "anchors": int(provenance["anchor_count"]),
                "included_episodes": int(provenance["included_episode_count"]),
                "included_frames": int(provenance["included_frame_count"]),
            }
        raise FileExistsError(f"refusing to replace incomplete output: {output_root}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent)
    )
    try:
        plans = plan_source_files(source_root, max_episodes=config.max_episodes)
        print(
            f"[prepare] selected {len(plans)} parquet files; "
            f"rows={sum(plan.rows for plan in plans)}",
            flush=True,
        )
        partial = config.max_episodes is not None
        selected_episode_max = _selected_episode_max(source_root, plans) if partial else -1
        copied_meta, episode_metadata_files = _copy_metadata(
            source_root,
            staging_root,
            selected_episode_max=selected_episode_max,
            partial=partial,
        )
        video_target = _create_video_symlink(
            source_root, staging_root, output_root, config.video_symlink
        )
        files = _convert_files(source_root, staging_root, plans)
        anchor_count, included, included_frames = _write_anchor_artifacts(
            staging_root, plans, config.non_idle, config.smoke_anchors, excluded
        )
        stats_work = staging_root / ".stats-work"
        stats = _compute_feature_stats(
            staging_root, plans, included_frames, stats_work, excluded
        )
        shutil.rmtree(stats_work)
        _update_metadata(
            staging_root,
            stats,
            partial,
            total_frames=sum(plan.rows for plan in plans),
            total_episodes=selected_episode_max + 1,
            anchor_count=anchor_count,
        )
        provenance = {
            "format_version": 1,
            "recipe_version": RECIPE_VERSION,
            "config": asdict(config),
            "config_fingerprint": fingerprint,
            "source_read_only": True,
            "source_identity": {
                "kind": "content_digest",
                "covers": "meta/** and data/**/*.parquet",
                "digest": config.source_digest,
            },
            "fps": EXPECTED_FPS,
            "state_action_definition": {
                "source_state": SOURCE_STATE_KEY,
                "source_action": SOURCE_ACTION_KEY,
                "renamed_to": [STATE_KEY, ACTION_KEY],
                "dim": STATE_DIM,
                "layout": "per arm [xyz(3), quat(4, w-first), gripper(1)]",
                "names": list(FEATURE_NAMES),
                "dtype": "float32",
                "rotation_encoding": "quaternion_wxyz",
                "quaternion_canonicalization": (
                    "per-episode sequential sign alignment of the state quaternion, "
                    "with the action quaternion tied to the state's hemisphere per frame"
                ),
                "dropped_source_columns": list(DROPPED_SOURCE_KEYS),
                "dropped_reason": (
                    "LeRobot maps every key starting with 'action' to a policy action "
                    "feature, so the extra action_* columns would collide; "
                    "action_eef_delta is additionally a componentwise subtraction whose "
                    "quaternion block is not a rotation difference"
                ),
            },
            "goal_definition": {
                "target_feature": STATE_KEY,
                "offset": HORIZON,
                "note": (
                    "state is itself the absolute EEF pose, so the goal is the LIBERO "
                    "arrangement: observation.state at t+H, with no separate feature"
                ),
            },
            "statistics_scope": stats["scope"],
            "anchor_definition": {
                "goal_offset": HORIZON,
                "action_window": f"[t:t+{HORIZON}]",
                "success_required": False,
                "excluded_episodes": list(config.exclude_episodes),
                "canonical_task_text_required": True,
                "same_episode_and_contiguous_required": True,
                "non_idle_rule": (
                    "segment the consecutive-command delta of the 16-D absolute EEF "
                    "action; drop long idle and short non-idle ranges; trim one horizon"
                ),
                "non_idle_parameters": asdict(config.non_idle),
            },
            "files": files,
            "copied_meta": copied_meta,
            "episode_metadata_files": episode_metadata_files,
            "videos_symlink_target": video_target,
            "anchor_count": anchor_count,
            "included_episode_count": len(included),
            "included_frame_count": included_frames,
            "artifact_sha256": _artifact_hashes(staging_root, files),
        }
        _json_dump(staging_root / "goal_pose_provenance.json", provenance)
        result = verify_dataset(staging_root, source_root=source_root, require_success=False)
        _json_dump(
            staging_root / "_SUCCESS.json",
            {
                "status": "complete",
                "config_fingerprint": fingerprint,
                "provenance_sha256": _sha256(staging_root / "goal_pose_provenance.json"),
            },
        )
        os.replace(staging_root, output_root)
        return result
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--source-digest",
        default="",
        help="Expected source content digest; computed from the source when omitted.",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--exclude-episodes",
        type=int,
        nargs="*",
        default=[],
        help="Episode indices to drop from training anchors (bad demonstrations).",
    )
    parser.add_argument("--video-symlink", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--smoke-anchors", type=int, default=128)
    parser.add_argument("--action-delta", type=float, default=5e-4)
    parser.add_argument("--min-idle-len", type=int, default=14)
    parser.add_argument("--min-non-idle-len", type=int, default=HORIZON + 1)
    parser.add_argument(
        "--print-source-digest",
        action="store_true",
        help="Print the source content digest and exit.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not build; strictly verify the completed output and source pairing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_root = args.source_root.resolve()
    if args.print_source_digest:
        print(source_content_digest(source_root))
        return

    non_idle = NonIdleConfig(
        action_delta=args.action_delta,
        min_idle_len=args.min_idle_len,
        min_non_idle_len=args.min_non_idle_len,
        trim_last_steps=HORIZON,
    )
    if args.verify_only:
        result = verify_dataset(args.output_root.resolve(), source_root=source_root)
        print(json.dumps(result, sort_keys=True))
        return

    digest = args.source_digest or source_content_digest(source_root)
    config = BuildConfig(
        source_root=str(source_root),
        output_root=str(args.output_root.resolve()),
        source_digest=digest,
        max_episodes=args.max_episodes,
        exclude_episodes=tuple(sorted(args.exclude_episodes)),
        video_symlink=args.video_symlink,
        smoke_anchors=args.smoke_anchors,
        horizon=HORIZON,
        non_idle=non_idle,
    )
    result = prepare_dataset(config)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
