#!/usr/bin/env python3
"""Action-following fit-check for a DROID Stage1 checkpoint, over full episodes.

Stage1 conditions its action expert on the REAL (ground-truth) goal pose --
it never predicts the goal itself. Its `goal_token_source="se3_encoder"`
*encodes* a given goal pose into context tokens for the action expert; there
is no goal-pose decoder (that's a Stage2-only capability, gated by
`enable_pose_reconstruction`, which Stage1 has off -- see viz_goal_pose.py's
docstring for that comparison instead).

For each sampled episode, every frame gets its OWN prediction: given that
frame's real state and real (receding) goal -- exactly as in training -- the
model predicts a chunk_size-step action chunk; only the FIRST predicted
action is used (teacher-forced, one-step-ahead), matching what the dataset's
own `action` column is for that frame. This is what actually answers "does
the model's per-step action match the demonstration": no open-loop chunk
replay, no jitter from re-using a single stale prediction over 1 second.

Outputs, per episode:
  - action_dims_ep<N>.png: one subplot per action dimension (q1..q7,
    gripper), GT vs predicted, over the full episode's timesteps.
  - clip_ep<N>.mp4 (+ contact sheet): current-EE / receding-goal / predicted-
    next-EE (forward-kinematics'd from the predicted action, see
    franka_fk.py, validated to ~1e-8m against DROID's own reported cartesian
    positions) projected onto the real camera frame, for every frame.

IMPORTANT: predictions are sampled via predict_goal_conditioned_chunk(), NOT
`policy.predict_action_chunk` -- the latter silently drops se3_encoder goal
tokens at inference (by design, for closed-loop deployment), which puts a
Stage1 checkpoint fully out of its training distribution and yields
uninformed samples (~0.65 rad mean error observed).

This script also works unmodified on Stage2 checkpoints. Verified by code
inspection: Stage2's goal_token_source="learnable_queries" makes
`_goal_token_embeddings` ignore `batch` entirely (it returns the learned
`goal_queries`, not an SE(3) encoding), so predict_goal_conditioned_chunk's
monkeypatch is a no-op for Stage2 -- it calls the exact same
`_generate_actions_from_inputs_with_rtc` that `predict_action_chunk` itself
calls once `_uses_policy_continuous_generation()` is true (which it is for
learnable_queries), with identical arguments. There is no Stage1-style
train/inference mismatch for Stage2.

Reuses viz_goal_pose.py for dataset access, camera projection, and the mp4
writer (see its docstring for the camera-intrinsics/extrinsics caveats).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import franka_fk  # noqa: E402
import viz_goal_pose as vg  # noqa: E402

ACTION_DIM_NAMES = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "gripper"]


def build_postprocessor(cfg, checkpoint: Path, dataset_stats):
    from lerobot.policies.factory import make_pre_post_processors

    _, postprocessor = make_pre_post_processors(cfg, pretrained_path=str(checkpoint), dataset_stats=dataset_stats)
    return postprocessor


def predict_goal_conditioned_chunk(policy, processed: dict, num_steps: int | None = None):
    """Sample a continuous action chunk WITH the SE(3) goal tokens injected.

    `predict_action_chunk` intentionally drops the se3_encoder goal conditioning
    (`_uses_policy_continuous_generation` excludes it because closed-loop
    deployment has no future goal pose), so on a Stage1 checkpoint it samples in
    a context the model never saw in training. Teacher-forced validation DOES
    have the true goal, so route through the policy-owned prefill/denoise path
    with `batch["goal_pose"]` supplied to `_goal_token_embeddings` -- the exact
    conditioning `forward()` uses during training.
    """
    import torch
    from contextlib import nullcontext

    from lerobot.policies.molmoact2.modeling_molmoact2 import _torch_dtype

    goal_pose = processed["goal_pose"]
    orig = policy._goal_token_embeddings
    policy._goal_token_embeddings = lambda batch, **kw: orig({"goal_pose": goal_pose, **batch}, **kw)
    try:
        model_inputs = policy._model_inputs(processed)
        device = next(policy.parameters()).device
        batch_size = int(next(iter(model_inputs.values())).shape[0])
        model_dtype = _torch_dtype(policy.config.model_dtype)
        autocast = (
            torch.autocast(device_type=device.type, dtype=model_dtype)
            if device.type in {"cuda", "cpu"} and model_dtype in {torch.bfloat16, torch.float16}
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            actions = policy._generate_actions_from_inputs_with_rtc(
                model_inputs=model_inputs,
                action_dim_is_pad=processed.get("action_dim_is_pad"),
                num_steps=num_steps,
                generator=policy._rollout_generator_for_inputs(processed, batch_size=batch_size, device=device),
                inference_delay=None,
                prev_chunk_left_over=None,
                execution_horizon=None,
            )
    finally:
        policy._goal_token_embeddings = orig
    action_dim = policy._output_action_dim(processed)
    return actions[:, : policy.config.n_action_steps, :action_dim].to(dtype=torch.float32)


def predict_next_actions(policy, preprocessor, postprocessor, items: list[dict], device: str) -> tuple[np.ndarray, dict]:
    """Teacher-forced one-step-ahead prediction: batch[i]'s FIRST predicted
    action, directly comparable to the dataset's own `action` at that frame."""
    pred, batch = predict_full_chunks(policy, preprocessor, postprocessor, items, device)
    return pred[:, 0, :], batch


def predict_full_chunks(policy, preprocessor, postprocessor, items: list[dict], device: str) -> tuple[np.ndarray, dict]:
    """Full postprocessed (B, chunk_size, action_dim) chunks, one sample each."""
    import torch
    from torch.utils.data import default_collate

    batch = default_collate(items)
    processed = vg.to_device(preprocessor(batch), device)
    with torch.inference_mode():
        chunk = predict_goal_conditioned_chunk(policy, processed)
    pred = postprocessor(chunk).float().cpu().numpy()
    return pred, batch


def process_episode(
    repo_id: str, dataset_root: Path, delta_timestamps: dict, episode_index: int,
    policy, preprocessor, postprocessor, device: str, batch_size: int, camera: str,
) -> dict:
    dataset = vg.build_episode_dataset(repo_id, dataset_root, delta_timestamps, episode_index)
    length = len(dataset)
    frame_indices = list(range(length))

    pred_actions, gt_actions, ee_pose, extrinsics, images = [], [], [], [], []
    for start in range(0, length, batch_size):
        chunk_idx = frame_indices[start : start + batch_size]
        items = vg.fetch_items(dataset, episode_index, chunk_idx)
        pred, batch = predict_next_actions(policy, preprocessor, postprocessor, items, device)
        pred_actions.append(pred)
        gt_actions.append(batch["action"][:, 0, :].numpy() if batch["action"].ndim == 3 else batch["action"].numpy())
        ee_pose.append(batch["observation.ee_pose"].numpy())
        extrinsics.append(batch[f"camera_extrinsics.{camera}"].numpy())
        img = batch[f"observation.images.{camera}"].numpy()
        images.append(img[:, 0] if img.ndim == 5 else img)
        print(f"[episode {episode_index}] {min(start + batch_size, length)}/{length}", flush=True)

    return {
        "episode_index": episode_index,
        "fps": dataset.meta.fps,
        "mode": "teacher-forced per-frame",
        "pred_actions": np.concatenate(pred_actions, axis=0),
        "gt_actions": np.concatenate(gt_actions, axis=0),
        "ee_pose": np.concatenate(ee_pose, axis=0),
        "extrinsics": np.concatenate(extrinsics, axis=0),
        "images": np.concatenate(images, axis=0),
    }


def process_episode_chunk_stitched(
    repo_id: str, dataset_root: Path, delta_timestamps: dict, episode_index: int,
    policy, preprocessor, postprocessor, device: str, batch_size: int,
) -> dict:
    """Deployment-style open loop: ONE sampled chunk per chunk_size frames, all
    of its steps plotted. Removes the per-frame independent-sample jitter of the
    teacher-forced mode (each 15-frame span comes from a single coherent draw)."""
    dataset = vg.build_episode_dataset(repo_id, dataset_root, delta_timestamps, episode_index)
    length = len(dataset)
    chunk_size = int(policy.config.chunk_size)
    anchors = list(range(0, length, chunk_size))

    pred = np.full((length, len(ACTION_DIM_NAMES)), np.nan, dtype=np.float32)
    gt = np.full_like(pred, np.nan)
    for start in range(0, len(anchors), batch_size):
        idx = anchors[start : start + batch_size]
        items = vg.fetch_items(dataset, episode_index, idx)
        chunks, batch = predict_full_chunks(policy, preprocessor, postprocessor, items, device)
        gt_chunks = batch["action"].numpy()
        for i, t0 in enumerate(idx):
            n = min(chunk_size, length - t0)  # trim the final chunk's episode-end padding
            pred[t0 : t0 + n] = chunks[i, :n]
            gt[t0 : t0 + n] = gt_chunks[i, :n]
        print(f"[episode {episode_index}] {min(start + batch_size, len(anchors))}/{len(anchors)} chunks", flush=True)

    return {
        "episode_index": episode_index,
        "fps": dataset.meta.fps,
        "mode": "chunk-stitched open-loop",
        "pred_actions": pred,
        "gt_actions": gt,
    }


def save_action_dim_plot(data: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt, pred = data["gt_actions"], data["pred_actions"]
    t = np.arange(len(gt))
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    for dim, (ax, name) in enumerate(zip(axes.flat, ACTION_DIM_NAMES, strict=True)):
        mae = float(np.mean(np.abs(pred[:, dim] - gt[:, dim])))
        ax.plot(t, gt[:, dim], label="GT", color="steelblue")
        ax.plot(t, pred[:, dim], label="pred", color="indianred", linestyle="--")
        ax.set_title(f"{name}  MAE={mae:.4f}")
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("frame")
    fig.suptitle(f"episode {data['episode_index']} -- action dims: GT vs predicted ({data['mode']})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_episode_clip(data: dict, intrinsics: tuple[float, float, float, float], goal_horizon: int, out_dir: Path) -> None:
    length = len(data["gt_actions"])
    pred_ee = franka_fk.panda_fk(data["pred_actions"][:, :7])
    ee_pose = data["ee_pose"]

    frames = []
    for t in range(length):
        t_base_cam = vg.cam2base_matrix(data["extrinsics"][t])
        goal_t = min(t + goal_horizon, length - 1)  # hold the last real pose once the receding goal runs past episode end
        err = float(np.linalg.norm(pred_ee[t] - ee_pose[t, 0, :3]))
        title = f"ep{data['episode_index']} frame {t}/{length} pred-vs-cur={err:.3f}m"
        frames.append(
            vg.draw_overlay(
                data["images"][t],
                current_xyz=ee_pose[t, 0, :3],
                gt_xyz=ee_pose[goal_t, 0, :3],
                pred_xyz=pred_ee[t],
                t_base_cam=t_base_cam,
                intrinsics=intrinsics,
                title=title,
            )
        )
    ep = data["episode_index"]
    vg.write_mp4_h264(frames, out_dir / f"clip_ep{ep}.mp4", fps=data["fps"])
    vg.save_grid(frames[:: max(1, length // 16)][:16], out_dir / f"clip_ep{ep}_sheet.png", cols=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Stage1 pretrained_model checkpoint dir")
    parser.add_argument("--dataset-root", type=Path, default=Path("/scratch/shailesh.xml/datasets/droid_1.0.1_goal_pose"))
    parser.add_argument("--repo-id", type=str, default="lerobot/droid_1.0.1")
    parser.add_argument("--camera", type=str, default="exterior_1_left",
                         choices=["exterior_1_left", "exterior_2_left", "wrist_left"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=4)
    parser.add_argument("--episode-indices", type=str, default=None,
                         help="comma-separated explicit episode indices to evaluate instead of "
                              "sampling from the training anchor manifest. Bypasses --num-episodes, "
                              "--seed, and --max-episode-length -- use this to evaluate episodes that "
                              "were never in valid_anchor_indices.parquet (e.g. a held-out probe set).")
    parser.add_argument("--max-episode-length", type=int, default=400, help="skip episodes longer than this")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--goal-horizon", type=int, default=15, help="frames ahead used as the visualized receding goal")
    parser.add_argument("--fx", type=float, default=vg.DEFAULT_FX)
    parser.add_argument("--fy", type=float, default=vg.DEFAULT_FY)
    parser.add_argument("--cx", type=float, default=vg.DEFAULT_CX)
    parser.add_argument("--cy", type=float, default=vg.DEFAULT_CY)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--skip-video", action="store_true", help="only write the action-dim plots + summary.csv")
    parser.add_argument("--chunk-stitched", action="store_true",
                         help="one sampled chunk per chunk_size frames, all steps plotted (implies --skip-video)")
    args = parser.parse_args()

    sys.path.insert(0, str(WS / "lerobot/src"))
    import lerobot.policies.factory  # noqa: F401  registers "molmoact2" etc. with PreTrainedConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps

    vg._ensure_ffmpeg_on_path()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    intrinsics = (args.fx, args.fy, args.cx, args.cy)

    ds_meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=str(args.dataset_root))
    cfg = PreTrainedConfig.from_pretrained(str(args.checkpoint))
    policy, preprocessor = vg.load_policy_and_preprocessor(
        args.checkpoint, args.device, dataset_root=args.dataset_root, repo_id=args.repo_id
    )
    postprocessor = build_postprocessor(cfg, args.checkpoint, ds_meta.stats)
    delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)

    if args.episode_indices is not None:
        episodes = [int(e) for e in args.episode_indices.split(",") if e.strip()]
        print(f"[viz] using {len(episodes)} explicit episode indices (manifest sampling bypassed)", flush=True)
    else:
        # Episode length from the dataset's own authoritative bounds, not the anchor manifest
        # (which excludes idle/invalid frames and would underestimate episode length).
        manifest = vg.read_anchor_manifest(args.dataset_root / "valid_anchor_indices.parquet")
        valid_episodes = manifest["episode_index"].unique()
        ep_meta = ds_meta.episodes
        lengths = np.asarray(ep_meta["dataset_to_index"]) - np.asarray(ep_meta["dataset_from_index"])
        short_enough = np.flatnonzero(lengths <= args.max_episode_length)
        candidates = np.intersect1d(short_enough, valid_episodes)
        print(f"[viz] {len(candidates)}/{len(valid_episodes)} valid episodes are <= {args.max_episode_length} frames", flush=True)
        episodes = rng.choice(candidates, size=min(args.num_episodes, len(candidates)), replace=False).tolist()

    summary_rows = []
    for episode_index in episodes:
        if args.chunk_stitched:
            data = process_episode_chunk_stitched(
                args.repo_id, args.dataset_root, delta_timestamps, int(episode_index),
                policy, preprocessor, postprocessor, args.device, args.batch_size,
            )
        else:
            data = process_episode(
                args.repo_id, args.dataset_root, delta_timestamps, int(episode_index),
                policy, preprocessor, postprocessor, args.device, args.batch_size, args.camera,
            )
        save_action_dim_plot(data, args.output_dir / f"action_dims_ep{episode_index}.png")
        if not args.skip_video and not args.chunk_stitched:
            render_episode_clip(data, intrinsics, args.goal_horizon, args.output_dir)
        dim_mae = np.mean(np.abs(data["pred_actions"] - data["gt_actions"]), axis=0)
        summary_rows.append(
            {"episode_index": int(episode_index), "num_frames": len(data["gt_actions"]),
             "mae_overall": float(dim_mae.mean()),
             **{f"mae_{name}": float(v) for name, v in zip(ACTION_DIM_NAMES, dim_mae, strict=True)}}
        )
        print(f"[episode {episode_index}] done, {len(data['gt_actions'])} frames, mean|action MAE|={dim_mae.mean():.4f}", flush=True)

    vg.write_csv(args.output_dir / "summary.csv", summary_rows)
    print(f"[viz] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
