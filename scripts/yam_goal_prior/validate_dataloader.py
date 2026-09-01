#!/usr/bin/env python3
"""Golden-check real YAM manifest batches through the MolmoAct2 processor.

Unlike the DROID equivalent this does not require a Stage-1 checkpoint: the
policy config is built directly so the check can run *before* the first training
launch. Pass ``--checkpoint`` to validate an existing Stage-1 save instead.

It answers three questions the plan flagged as open:

1. Does the processor's quantile normalization match a hand-written reference,
   with both bimanual gripper dimensions passed through raw?
2. Is the ``goal_pose`` time axis actually present? If it is not,
   ``_extract_goal_pose`` returns ``(None, None)`` and Stage 2 silently drops
   ``L_pose`` with no warning -- the single most dangerous failure mode here.
3. **What is the real tokenized sequence length with three cameras at
   360-480p?** ``MOLMOACT2_IMAGE_TOKENS_PER_IMAGE = 196`` is an assumption; a
   larger input may produce more crops, and the processor's oversize check is a
   hard ValueError with no truncation path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType

# The goal target IS observation.state at t+H -- the LIBERO arrangement, where
# one tensor carries a time axis and the processor slices [:, 0] for the current
# state and [:, -1] for the goal.
GOAL_KEY = OBS_STATE
IMAGE_KEYS = [
    "observation.images.top",
    "observation.images.left",
    "observation.images.right",
]
STATE_DIM = 16
GOAL_DIM = 16


def _gripper_indices(dataset_meta, key: str) -> list[int]:
    names = dataset_meta.features.get(key, {}).get("names")
    if isinstance(names, dict):
        flat: list[str] = []
        for value in names.values():
            flat.extend(value if isinstance(value, list) else [value])
        names = flat
    if not isinstance(names, list):
        raise RuntimeError(f"dataset feature {key!r} has no per-dimension names")
    indices = [i for i, name in enumerate(names) if "gripper" in str(name).lower()]
    if len(indices) != 2:
        raise RuntimeError(f"{key} must declare exactly two gripper dims, got {indices}")
    return indices


def _expected_normalized(
    values: torch.Tensor, stats: dict[str, Any], *, gripper_indices: list[int]
) -> torch.Tensor:
    q01 = torch.as_tensor(np.asarray(stats["q01"]), dtype=values.dtype)
    q99 = torch.as_tensor(np.asarray(stats["q99"]), dtype=values.dtype)
    normalized = 2.0 * (values - q01) / (q99 - q01) - 1.0
    for index in gripper_indices:
        normalized[..., index] = values[..., index]
    return normalized.clamp(-1.0, 1.0)


def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.to(torch.float64) - expected.to(torch.float64)).abs().max())


def _build_config(
    dataset_root: Path,
    repo_id: str,
    manifest: Path,
    checkpoint_path: Path,
    horizon: int,
    max_sequence_length: int,
) -> TrainPipelineConfig:
    policy = MolmoAct2Config(
        chunk_size=horizon,
        n_action_steps=horizon,
        action_mode="continuous",
        setup_type="bimanual yam robot arms",
        control_mode="absolute joint pose",
        image_keys=list(IMAGE_KEYS),
        max_sequence_length=max_sequence_length,
        normalize_gripper=False,
        checkpoint_path=str(checkpoint_path),
        device="cpu",
        # Stage-2 topology: this is the configuration whose goal-pose plumbing
        # and image handling need validating.
        enable_goal_pose=True,
        goal_pose_feature_key=GOAL_KEY,
        goal_token_source="learnable_queries",
        goal_conditioning_mode="semantic_visual_recurrent",
        target_pose_delta_index=horizon,
        mask_image_from_action_expert=True,
        enable_pose_reconstruction=True,
    )
    dataset = DatasetConfig(
        repo_id=repo_id,
        root=str(dataset_root),
        sample_indices_path=str(manifest),
        video_backend="pyav",
    )
    cfg = TrainPipelineConfig(dataset=dataset, policy=policy)
    cfg.dataset.image_transforms.enable = False
    return cfg


def validate(
    dataset_root: Path,
    repo_id: str,
    checkpoint_path: Path,
    horizon: int,
    max_sequence_length: int,
    sample_positions: list[int] | None = None,
) -> dict[str, Any]:
    manifest_path = dataset_root / "valid_anchor_indices.parquet"
    cfg = _build_config(
        dataset_root, repo_id, manifest_path, checkpoint_path, horizon, max_sequence_length
    )
    dataset = make_dataset(cfg)
    if len(dataset) < 3:
        raise ValueError("golden validation requires at least three manifest anchors")

    positions = sample_positions or [0, len(dataset) // 2, len(dataset) - 1]
    if len(set(positions)) != len(positions) or any(
        position < 0 or position >= len(dataset) for position in positions
    ):
        raise ValueError(f"invalid manifest positions: {positions}")

    # Mirror lerobot.policies.factory.make_policy's feature wiring without
    # instantiating the 21 GB model: only the processor is under test.
    features = dataset_to_policy_features(dataset.meta.features)
    policy = cfg.policy
    policy.output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    policy.input_features = {k: f for k, f in features.items() if k not in policy.output_features}
    policy.set_dataset_feature_metadata(dataset.meta.features)

    manifest = dataset.reader._sample_manifest_rows
    if manifest is None:
        raise RuntimeError("dataset did not load the configured sample manifest")
    items = [dataset[position] for position in positions]
    for position, item in zip(positions, items, strict=True):
        expected_identity = (
            int(manifest.indices[position]),
            int(manifest.episode_indices[position]),
            int(manifest.frame_indices[position]),
        )
        actual_identity = (
            int(item["index"]),
            int(item["episode_index"]),
            int(item["frame_index"]),
        )
        if actual_identity != expected_identity:
            raise RuntimeError(
                f"manifest identity mismatch at position {position}: "
                f"{actual_identity} != {expected_identity}"
            )

    batch = default_collate(items)
    for camera_key in dataset.meta.camera_keys:
        if batch[camera_key].dtype == torch.uint8:
            batch[camera_key] = batch[camera_key].to(dtype=torch.float32) / 255.0

    if batch[GOAL_KEY].ndim != 3 or batch[GOAL_KEY].shape[1] < 2:
        raise RuntimeError(
            f"{GOAL_KEY} arrived with shape {list(batch[GOAL_KEY].shape)}; the goal time axis "
            "is missing, so _extract_goal_pose would return None and Stage 2 would silently "
            "drop the pose reconstruction loss"
        )

    # Shared-key layout: the tensor keeps its time axis through the normalizer,
    # index 0 being the current frame and index -1 the t+H goal.
    raw_state_series = batch[OBS_STATE].clone()
    raw_state = raw_state_series[:, 0].clone()
    raw_action = batch[ACTION].clone()
    raw_goal = raw_state_series[:, -1].clone()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.input_features, **policy.output_features},
                "norm_map": policy.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.output_features,
                "norm_map": policy.normalization_mapping,
            }
        },
    )
    processed = preprocessor(batch)

    state_grippers = _gripper_indices(dataset.meta, OBS_STATE)
    action_grippers = _gripper_indices(dataset.meta, ACTION)
    goal_grippers = state_grippers  # same feature
    expected_state_series = _expected_normalized(
        raw_state_series, dataset.meta.stats[OBS_STATE], gripper_indices=state_grippers
    )
    errors = {
        "state": _max_error(processed[OBS_STATE], expected_state_series),
        "action": _max_error(
            processed[ACTION][..., :STATE_DIM],
            _expected_normalized(
                raw_action, dataset.meta.stats[ACTION], gripper_indices=action_grippers
            ),
        ),
        "goal_pose": _max_error(processed["goal_pose"], expected_state_series[:, -1]),
    }
    if any(error > 1e-6 for error in errors.values()):
        raise RuntimeError(f"processor normalization mismatch: {errors}")

    if bool(processed["action_horizon_is_pad"].any()):
        raise RuntimeError("manifest batch unexpectedly contains padded action rows")
    if bool(processed["goal_pose_is_pad"].any()):
        raise RuntimeError(f"manifest batch unexpectedly contains a padded t+{horizon} goal")
    # The goal must actually differ from the current state, otherwise the time
    # axis collapsed and Stage 1 would be conditioned on the present.
    if torch.allclose(processed["goal_pose"], processed[OBS_STATE][:, 0], atol=1e-6):
        raise RuntimeError("goal_pose equals the current state; the t+H offset was lost")
    expected_action_shape = (len(positions), horizon, 32)
    if tuple(processed[ACTION].shape) != expected_action_shape:
        raise RuntimeError(
            f"action shape {tuple(processed[ACTION].shape)} != {expected_action_shape}"
        )
    if not bool(processed["action_dim_is_pad"][:, :STATE_DIM].logical_not().all()):
        raise RuntimeError("real YAM action dimensions were marked as padding")
    if not bool(processed["action_dim_is_pad"][:, STATE_DIM:].all()):
        raise RuntimeError("padded MolmoAct2 action dimensions were marked as real")

    visual_keys = {
        key: list(value.shape)
        for key, value in processed.items()
        if key in {"pixel_values", "image_token_pooling", "image_grids", "image_num_crops"}
        and torch.is_tensor(value)
    }
    if not visual_keys:
        raise RuntimeError("processed batch contains no visual model inputs")
    for key in ("input_ids", "attention_mask", "goal_pose"):
        if key not in processed or not torch.isfinite(processed[key]).all():
            raise RuntimeError(f"processed batch has invalid {key}")

    sequence_length = int(processed["input_ids"].shape[1])
    headroom = max_sequence_length - sequence_length
    if headroom < 0:
        raise RuntimeError(
            f"tokenized sequence length {sequence_length} exceeds "
            f"max_sequence_length={max_sequence_length}"
        )

    return {
        "status": "passed",
        "dataset_frames": dataset.meta.total_frames,
        "manifest_anchors": len(dataset),
        "manifest_episodes": dataset.num_episodes,
        "positions": positions,
        "absolute_indices": [int(manifest.indices[p]) for p in positions],
        "episode_indices": [int(manifest.episode_indices[p]) for p in positions],
        "frame_indices": [int(manifest.frame_indices[p]) for p in positions],
        "raw_shapes": {
            "state": list(raw_state.shape),
            "action": list(raw_action.shape),
            "goal_feature": list(batch[GOAL_KEY].shape),
        },
        "goal_vs_current_state_l2": float(
            torch.linalg.norm(
                (processed["goal_pose"] - processed[OBS_STATE][:, 0]).to(torch.float64), dim=-1
            ).mean()
        ),
        "processed_shapes": {
            "state": list(processed[OBS_STATE].shape),
            "action": list(processed[ACTION].shape),
            "goal_pose": list(processed["goal_pose"].shape),
        },
        "visual_inputs": visual_keys,
        "max_abs_error": errors,
        "raw_gripper_indices": {
            OBS_STATE: state_grippers,
            ACTION: action_grippers,
            GOAL_KEY: goal_grippers,
        },
        "sequence_length": sequence_length,
        "max_sequence_length": max_sequence_length,
        "sequence_headroom": headroom,
        "camera_shapes": {
            key: list(batch[key].shape[-3:]) for key in IMAGE_KEYS if key in batch
        },
    }


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "real_robot_datasets/yam_3task_goal_pose",
    )
    parser.add_argument("--repo-id", default="local/yam_blocks_goal_pose")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path.home() / "checkpoints/MolmoAct2",
        help="MolmoAct2 snapshot supplying the tokenizer/image processor.",
    )
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--max-sequence-length", type=int, default=896)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = validate(
        args.dataset_root.expanduser().resolve(),
        args.repo_id,
        args.checkpoint_path.expanduser().resolve(),
        args.horizon,
        args.max_sequence_length,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    print(
        f"[validate] sequence_length={report['sequence_length']} "
        f"max={report['max_sequence_length']} headroom={report['sequence_headroom']}"
    )
    print(f"[validate] {report['status'].upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
