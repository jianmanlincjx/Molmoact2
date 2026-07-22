#!/usr/bin/env python3
"""Validate four MolmoAct2 LIBERO smoke paths and initialization continuity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open


AUDIT_RE = re.compile(r"Policy training audit:\s*(\{.*\})")
STEP_RE = re.compile(r"step:(\d+).*?loss:([0-9.eE+-]+)")


def action_expert_fingerprint(checkpoint: Path) -> str:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]
    samples: dict[str, tuple[tuple[int, ...], bytes]] = {}
    for shard_name in shards:
        with safe_open(checkpoint / shard_name, framework="pt", device="cpu") as shard:
            for full_name in shard.keys():
                if "action_expert." not in full_name:
                    continue
                name = full_name.split("action_expert.", 1)[1]
                tensor = shard.get_tensor(full_name).reshape(-1)
                sample = torch.cat((tensor[:16], tensor[-16:])).to(dtype=torch.float32)
                samples[name] = (tuple(shard.get_slice(full_name).get_shape()), sample.numpy().tobytes())
    if not samples:
        raise ValueError(f"No action-expert tensors found in {checkpoint}")
    digest = hashlib.sha256()
    for name in sorted(samples):
        shape, payload = samples[name]
        digest.update(name.encode())
        digest.update(str(shape).encode())
        digest.update(payload)
    return digest.hexdigest()


def loaded_action_expert_fingerprint(checkpoint: Path) -> str:
    """Use module state order so the result matches the runtime training audit."""
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy, _module_fingerprint

    config = PreTrainedConfig.from_pretrained(
        checkpoint,
        cli_overrides=[
            "--device=cpu",
            "--disable_visual_input=false",
            "--train_action_expert_only=false",
        ],
    )
    policy = MolmoAct2Policy.from_pretrained(checkpoint, config=config)
    return _module_fingerprint(policy._action_expert())


def load_run(smoke_root: Path, name: str, expected_steps: int) -> dict:
    log_path = smoke_root / f"{name}.log"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    audits = [json.loads(match.group(1)) for match in AUDIT_RE.finditer(text)]
    if not audits:
        raise ValueError(f"No policy training audit in {log_path}")
    points = [(int(step), float(loss)) for step, loss in STEP_RE.findall(text)]
    if not points or points[-1][0] < expected_steps:
        raise ValueError(f"{name} stopped before step {expected_steps}: {points[-1:]}")
    if not all(math.isfinite(loss) for _, loss in points):
        raise ValueError(f"{name} produced non-finite loss")
    checkpoint = smoke_root / name / "checkpoints" / "last" / "pretrained_model"
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    return {
        "initial_audit": audits[-1],
        "last_step": points[-1][0],
        "last_loss": points[-1][1],
        "checkpoint": str(checkpoint),
        "final_action_expert_fingerprint": action_expert_fingerprint(checkpoint),
        "saved_disable_visual_input": config["disable_visual_input"],
        "saved_train_action_expert_only": config["train_action_expert_only"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=50)
    args = parser.parse_args()

    results = {
        name: load_run(args.smoke_root, name, args.expected_steps)
        for name in (
            "01_baseline_50step",
            "02_stage1_50step",
            "03_stage2_from_stage1_50step",
        )
    }
    canonical_audit = json.loads(
        (args.canonical_dir / "bootstrap_audit.json").read_text(encoding="utf-8")
    )
    canonical_hash = canonical_audit["action_expert_saved_fingerprint"]
    for name in ("01_baseline_50step", "02_stage1_50step"):
        observed = results[name]["initial_audit"]["action_expert_fingerprint"]
        if observed != canonical_hash:
            raise ValueError(f"{name} did not start from canonical action expert: {observed} != {canonical_hash}")

    stage2_initial = results["03_stage2_from_stage1_50step"]["initial_audit"][
        "action_expert_fingerprint"
    ]
    stage1_checkpoint = Path(results["02_stage1_50step"]["checkpoint"])
    stage1_runtime_hash = loaded_action_expert_fingerprint(stage1_checkpoint)
    results["02_stage1_50step"]["final_runtime_action_expert_fingerprint"] = stage1_runtime_hash
    if stage2_initial != stage1_runtime_hash:
        raise ValueError("Stage 2 did not inherit the final Stage-1 action-expert weights.")

    expected_groups = {
        "01_baseline_50step": {"vlm", "vit", "connector", "action_expert"},
        "02_stage1_50step": {"action_expert"},
        "03_stage2_from_stage1_50step": {"vlm", "vit", "connector", "action_expert"},
    }
    expected_visual = {
        "01_baseline_50step": False,
        "02_stage1_50step": True,
        "03_stage2_from_stage1_50step": False,
    }
    for name, groups in expected_groups.items():
        observed_groups = set(results[name]["initial_audit"]["trainable_parameter_groups"])
        if observed_groups != groups:
            raise ValueError(f"{name} parameter groups mismatch: {observed_groups} != {groups}")
        if results[name]["saved_disable_visual_input"] is not expected_visual[name]:
            raise ValueError(f"{name} saved the wrong visual-input state")

    report = {
        "status": "passed",
        "canonical_action_expert_fingerprint": canonical_hash,
        "runs": results,
    }
    report_path = args.smoke_root / "smoke_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[smoke] validation passed: {report_path}")


if __name__ == "__main__":
    main()
