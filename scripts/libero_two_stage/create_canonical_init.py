#!/usr/bin/env python3
"""Create the shared Molmo2-ER + random-action-expert initialization checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import _module_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="local/libero_lerobot_format")
    parser.add_argument("--molmoact2-checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite canonical checkpoint: {args.output_dir}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    config = MolmoAct2Config(
        checkpoint_path=str(args.molmoact2_checkpoint),
        vlm_checkpoint_path=str(args.vlm_checkpoint),
        randomize_action_expert=True,
        audit_bootstrap=True,
        device="cpu",
        model_dtype="bfloat16",
        action_mode="continuous",
        chunk_size=10,
        n_action_steps=10,
        setup_type="single franka robotic arm in libero",
        control_mode="delta end-effector pose",
        image_keys=["observation.images.image", "observation.images.image2"],
        gradient_checkpointing=True,
        freeze_embedding=True,
        normalize_gripper=False,
        enable_knowledge_insulation=False,
        push_to_hub=False,
    )
    policy = make_policy(config, ds_meta=metadata)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        dataset_stats=metadata.stats,
        dataset_meta=metadata,
    )

    policy.save_pretrained(args.output_dir)
    preprocessor.save_pretrained(args.output_dir)
    postprocessor.save_pretrained(args.output_dir)

    trainable = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in policy.parameters())
    audit = dict(policy.bootstrap_audit)
    audit.update(
        {
            "seed": args.seed,
            "action_expert_saved_fingerprint": _module_fingerprint(policy._action_expert()),
            "trainable_parameters": trainable,
            "total_parameters": total,
            "dataset_repo_id": args.dataset_repo_id,
            "dataset_root": str(args.dataset_root.resolve()),
        }
    )
    (args.output_dir / "bootstrap_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[canonical] saved {args.output_dir}")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
