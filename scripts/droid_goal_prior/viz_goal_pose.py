#!/usr/bin/env python3
"""Qualitative fit-check for a DROID goal-pose checkpoint (Stage1 or Stage2).

Runs the policy over its own training anchors ("does the model fit the data
it trained on") and renders the predicted goal pose against the ground-truth
goal pose, projected onto the real external camera frame:

  - MP4 clips (+ PNG contact-sheet fallback) of a few episodes with per-frame
    current-EE / GT-goal / predicted-goal markers.
  - Error histograms and a pred-vs-GT scatter over a larger sample.

Without --checkpoint, runs in smoke-test mode: no policy is loaded, only the
dataset access pattern and camera projection are exercised (current-EE vs GT,
no prediction) -- useful for validating the pipeline before a checkpoint
exists.

Dataset access is restricted to the handful of episodes actually sampled
(``LeRobotDataset(..., episodes=[...])``) rather than the full ~17.3M-row
dataset: an unrestricted load forces a full parquet->Arrow cache build (only
worth paying once, at training time), which is unnecessary for sampling a few
hundred frames here.

Camera projection uses the per-frame ``camera_extrinsics.<cam>`` feature
(cam2base: xyz translation + extrinsic XYZ-Euler roll/pitch/yaw -- see
https://huggingface.co/KarlP/droid) inverted to base->cam, composed with a
FIXED approximate pinhole intrinsic. This dataset ships no per-episode
intrinsics (DROID's real ``intrinsics.json`` is keyed by an original-episode
UUID that isn't retained after conversion), so --fx/--fy/--cx/--cy are a
scaled-down approximation of typical ZED2 factory values. The current-EE
marker is always drawn as a built-in sanity check: it's a known-correct 3D
point, so if it visibly misses the real gripper in the video, the intrinsics
approximation needs adjusting.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]


def _ensure_ffmpeg_on_path() -> None:
    """write_mp4_h264 shells out to the literal command `ffmpeg`; no system ffmpeg is
    installed on this cluster. imageio-ffmpeg ships a static binary but under a
    versioned filename (e.g. ffmpeg-linux-x86_64-v7.0.2), which `which ffmpeg` won't
    match -- symlink it to the expected name once, then prepend that dir to PATH."""
    import shutil

    if shutil.which("ffmpeg") is not None:
        return
    import imageio_ffmpeg

    bin_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    link = bin_dir / "ffmpeg"
    if not link.exists():
        link.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _load_libero_viz():
    """Reuse the generic, embodiment-agnostic helpers from the LIBERO viz script
    instead of duplicating them (denormalization, drawing primitives, grid/CSV/mp4
    writers, and the goal-pose-decoder inference call, which is config-driven and
    already handles both the Stage1 and Stage2 conditioning branches)."""
    path = WS / "scripts/libero_goal_prior/viz_goal_pose.py"
    spec = importlib.util.spec_from_file_location("libero_viz_goal_pose", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses in the module need this to resolve
    spec.loader.exec_module(module)
    return module


_libero = _load_libero_viz()
denormalize_state = _libero.denormalize_state
to_hwc_uint8 = _libero.to_hwc_uint8
draw_marker = _libero.draw_marker
save_grid = _libero.save_grid
write_csv = _libero.write_csv
write_mp4_h264 = _libero.write_mp4_h264
predict_goal_pose_from_policy = _libero.predict_goal_pose_from_policy
load_policy_and_preprocessor = _libero.load_policy_and_preprocessor

COLOR_CURRENT = (255, 90, 40)
COLOR_GT = (40, 200, 60)
COLOR_PRED = (40, 40, 255)

# Approximate pinhole intrinsics for the 320x180 exterior frames (see module
# docstring for why this is a fixed approximation rather than per-episode
# values). Principal point at image center; focal length scaled down from a
# typical ZED2 factory 2K calibration. Override via CLI if you have real ones.
DEFAULT_FX = 283.0
DEFAULT_FY = 283.0
DEFAULT_CX = 160.0
DEFAULT_CY = 90.0


def cam2base_matrix(extrinsics: np.ndarray) -> np.ndarray:
    """[x,y,z,roll,pitch,yaw] (cam2base, extrinsic XYZ-Euler) -> 4x4 T_base_cam."""
    from scipy.spatial.transform import Rotation

    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", extrinsics[3:6]).as_matrix()
    transform[:3, 3] = extrinsics[:3]
    return transform


def project_world_to_pixel(
    xyz_base: np.ndarray,
    t_base_cam: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int] | None:
    """Project a robot-base-frame point to pixel (u, v) via a pinhole camera model."""
    fx, fy, cx, cy = intrinsics
    rot, trans = t_base_cam[:3, :3], t_base_cam[:3, 3]
    p_cam = rot.T @ (np.asarray(xyz_base, dtype=np.float64) - trans)
    if p_cam[2] <= 1e-3:  # behind or on the camera plane
        return None
    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy
    if not (0 <= u < width and 0 <= v < height):
        return None
    return int(round(u)), int(round(v))


def draw_overlay(
    image: np.ndarray,
    *,
    current_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    pred_xyz: np.ndarray | None,
    t_base_cam: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    title: str = "",
) -> np.ndarray:
    import cv2

    image_bgr = cv2.cvtColor(to_hwc_uint8(image), cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    for xyz, color, label in (
        (current_xyz, COLOR_CURRENT, "cur"),
        (gt_xyz, COLOR_GT, "gt"),
        (pred_xyz, COLOR_PRED, "pred"),
    ):
        if xyz is None:
            continue
        uv = project_world_to_pixel(xyz, t_base_cam, intrinsics, w, h)
        draw_marker(image_bgr, uv, color, label)
    if title:
        cv2.putText(
            image_bgr, title[:100], (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_stat_plots(rows: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    err3d = np.asarray([r["err3d"] for r in rows])
    axis_err = np.asarray([[r["err_x"], r["err_y"], r["err_z"]] for r in rows])
    pred_xyz = np.asarray([r["_pred_xyz"] for r in rows])
    gt_xyz = np.asarray([r["_gt_xyz"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes[0, 0].hist(err3d, bins=30, color="steelblue")
    axes[0, 0].set_title(f"3D goal-pose error (m)  mean={err3d.mean():.3f} median={np.median(err3d):.3f}")
    for i, name in enumerate(("x", "y", "z")):
        ax = axes.flat[i + 1]
        ax.hist(axis_err[:, i], bins=30, color="indianred")
        ax.set_title(f"|{name}| error (m)  mean={axis_err[:, i].mean():.3f}")
    fig.tight_layout()
    fig.savefig(out_dir / "error_hist.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, name in enumerate(("x", "y", "z")):
        ax = axes[i]
        ax.scatter(gt_xyz[:, i], pred_xyz[:, i], s=8, alpha=0.5)
        lo, hi = gt_xyz[:, i].min(), gt_xyz[:, i].max()
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel(f"GT {name} (m)")
        ax.set_ylabel(f"pred {name} (m)")
    fig.tight_layout()
    fig.savefig(out_dir / "pred_vs_gt_scatter.png", dpi=120)
    plt.close(fig)


def load_ee_pose_quantiles(dataset_root: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = json.loads((dataset_root / "meta/stats.json").read_text())["observation.ee_pose"]
    return np.asarray(stats["q01"], dtype=np.float64), np.asarray(stats["q99"], dtype=np.float64)


def read_anchor_manifest(manifest_path: Path):
    """Read (index, episode_index, frame_index) for every valid training anchor.

    This is a plain parquet read (a few hundred MB, seconds), not a dataset
    load -- cheap, and gives us everything needed to pick episodes/frames
    before touching any video/tensor data.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(manifest_path, columns=["index", "episode_index", "frame_index"])
    return table.to_pandas()


def pick_episode_groups(
    manifest, rng: np.random.Generator, num_episodes: int, samples_per_episode: int
) -> list[tuple[int, list[int]]]:
    """Pick distinct episodes and up to `samples_per_episode` anchor frame_indices each."""
    episodes = manifest["episode_index"].unique()
    chosen = rng.choice(episodes, size=min(num_episodes, len(episodes)), replace=False)
    groups = []
    for ep in chosen.tolist():
        frames = manifest.loc[manifest["episode_index"] == ep, "frame_index"].to_numpy()
        n = min(samples_per_episode, len(frames))
        groups.append((int(ep), sorted(rng.choice(frames, size=n, replace=False).tolist())))
    return groups


def build_episode_dataset(repo_id: str, dataset_root: Path, delta_timestamps: dict, episode_index: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        repo_id,
        root=str(dataset_root),
        delta_timestamps=delta_timestamps,
        episodes=[episode_index],
        video_backend="pyav",
        return_uint8=True,
    )


def fetch_items(dataset, episode_index: int, frame_indices: list[int]) -> list[dict]:
    """Frame `frame_index` is 0-based within an episode, matching the manifest's
    `frame_index` column: restricting a LeRobotDataset to one episode makes its
    local index directly equal to frame_index (see dataset_reader.get_item's
    docstring: idx is relative to the episode-filtered dataset)."""
    items = [dataset[i] for i in frame_indices]
    for item, frame_index in zip(items, frame_indices, strict=True):
        if int(item["episode_index"]) != episode_index or int(item["frame_index"]) != frame_index:
            raise RuntimeError(
                f"episode-local indexing assumption broke: requested (ep={episode_index}, "
                f"frame={frame_index}), got (ep={int(item['episode_index'])}, "
                f"frame={int(item['frame_index'])})"
            )
    return items


def to_device(batch: dict, device: str) -> dict:
    import torch

    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_goal_pose(policy, preprocessor, items: list[dict], device: str) -> tuple[np.ndarray, dict]:
    """Batch items through the preprocessor and decode the normalized goal pose."""
    from torch.utils.data import default_collate

    batch = default_collate(items)
    processed = to_device(preprocessor(batch), device)
    pred_norm = predict_goal_pose_from_policy(policy, processed)
    return pred_norm, batch


def run_stats_pass(
    repo_id: str,
    dataset_root: Path,
    delta_timestamps: dict,
    groups: list[tuple[int, list[int]]],
    policy,
    preprocessor,
    q01: np.ndarray,
    q99: np.ndarray,
    device: str,
) -> list[dict]:
    rows: list[dict] = []
    for episode_index, frame_indices in groups:
        dataset = build_episode_dataset(repo_id, dataset_root, delta_timestamps, episode_index)
        items = fetch_items(dataset, episode_index, frame_indices)
        pred_norm, batch = run_goal_pose(policy, preprocessor, items, device)
        pred = denormalize_state(pred_norm, q01, q99)
        ee = batch["observation.ee_pose"].numpy()
        for b, frame_index in enumerate(frame_indices):
            gt, cur = ee[b, -1], ee[b, 0]
            err = pred[b, :3] - gt[:3]
            rows.append(
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "current_xyz": " ".join(f"{x:.4f}" for x in cur[:3]),
                    "gt_xyz": " ".join(f"{x:.4f}" for x in gt[:3]),
                    "pred_xyz": " ".join(f"{x:.4f}" for x in pred[b, :3]),
                    "err_x": abs(float(err[0])),
                    "err_y": abs(float(err[1])),
                    "err_z": abs(float(err[2])),
                    "err3d": float(np.linalg.norm(err)),
                    "_pred_xyz": pred[b, :3].tolist(),
                    "_gt_xyz": gt[:3].tolist(),
                }
            )
        print(f"[stats] episode {episode_index}: {len(frame_indices)} samples", flush=True)
    return rows


def render_clips(
    repo_id: str,
    dataset_root: Path,
    delta_timestamps: dict,
    clip_episodes: list[int],
    policy,
    preprocessor,
    q01: np.ndarray,
    q99: np.ndarray,
    device: str,
    camera: str,
    intrinsics: tuple[float, float, float, float],
    clip_length: int,
    out_dir: Path,
) -> None:
    cam_key = f"observation.images.{camera}"
    extr_key = f"camera_extrinsics.{camera}"

    for made, episode_index in enumerate(clip_episodes):
        dataset = build_episode_dataset(repo_id, dataset_root, delta_timestamps, episode_index)
        length = min(clip_length, len(dataset))
        frame_indices = list(range(length))
        items = fetch_items(dataset, episode_index, frame_indices)

        if policy is not None:
            pred_norm, batch = run_goal_pose(policy, preprocessor, items, device)
            pred = denormalize_state(pred_norm, q01, q99)
        else:
            from torch.utils.data import default_collate

            batch = default_collate(items)
            pred = None

        ee = batch["observation.ee_pose"].numpy()
        extrinsics = batch[extr_key].numpy()
        images = batch[cam_key].numpy()
        if images.ndim == 5:  # squeeze a size-1 leading time axis, if present
            images = images[:, 0]

        frames = []
        for t in range(length):
            t_base_cam = cam2base_matrix(extrinsics[t])
            pred_xyz = pred[t, :3] if pred is not None else None
            title = f"ep{episode_index} frame {t}"
            if pred_xyz is not None:
                title += f" err3d={float(np.linalg.norm(pred_xyz - ee[t, -1, :3])):.3f}m"
            frames.append(
                draw_overlay(
                    images[t],
                    current_xyz=ee[t, 0, :3],
                    gt_xyz=ee[t, -1, :3],
                    pred_xyz=pred_xyz,
                    t_base_cam=t_base_cam,
                    intrinsics=intrinsics,
                    title=title,
                )
            )

        write_mp4_h264(frames, out_dir / f"clip_{made:02d}_ep{episode_index}.mp4", fps=dataset.meta.fps)
        save_grid(frames[:16], out_dir / f"clip_{made:02d}_ep{episode_index}_sheet.png", cols=4)
        print(f"[clip] {made + 1}/{len(clip_episodes)} episode {episode_index} ({length} frames)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=None,
                         help="pretrained_model checkpoint dir (Stage1 or Stage2). Omit for a smoke test (no policy).")
    parser.add_argument("--dataset-root", type=Path, default=Path("/scratch/shailesh.xml/datasets/droid_1.0.1_goal_pose"))
    parser.add_argument("--repo-id", type=str, default="lerobot/droid_1.0.1")
    parser.add_argument("--camera", type=str, default="exterior_1_left",
                         choices=["exterior_1_left", "exterior_2_left", "wrist_left"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-stat-episodes", type=int, default=20)
    parser.add_argument("--samples-per-episode", type=int, default=15)
    parser.add_argument("--num-clips", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=45, help="frames per clip (~3s at 15fps)")
    parser.add_argument("--fx", type=float, default=DEFAULT_FX)
    parser.add_argument("--fy", type=float, default=DEFAULT_FY)
    parser.add_argument("--cx", type=float, default=DEFAULT_CX)
    parser.add_argument("--cy", type=float, default=DEFAULT_CY)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    sys.path.insert(0, str(WS / "lerobot/src"))
    import lerobot.policies.factory  # noqa: F401  registers "molmoact2" etc. with PreTrainedConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps

    _ensure_ffmpeg_on_path()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    intrinsics = (args.fx, args.fy, args.cx, args.cy)

    ds_meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=str(args.dataset_root))
    cfg = policy = preprocessor = None
    if args.checkpoint is not None:
        cfg = PreTrainedConfig.from_pretrained(str(args.checkpoint))
        policy, preprocessor = load_policy_and_preprocessor(
            args.checkpoint, args.device, dataset_root=args.dataset_root, repo_id=args.repo_id
        )
    else:
        print("[viz] no --checkpoint given: smoke-test mode (dataset + projection only, no prediction)")
    # target_pose_delta_index=15 matches both stages' training config; only needed
    # here to fall back to the same window when running the checkpoint-free smoke test.
    delta_timestamps = (
        resolve_delta_timestamps(cfg, ds_meta) if cfg is not None else {"observation.ee_pose": [0.0, 15.0 / ds_meta.fps]}
    )
    q01, q99 = load_ee_pose_quantiles(args.dataset_root)

    manifest = read_anchor_manifest(args.dataset_root / "valid_anchor_indices.parquet")
    print(f"[viz] {len(manifest)} valid anchors across {manifest['episode_index'].nunique()} episodes", flush=True)

    groups = pick_episode_groups(manifest, rng, args.num_stat_episodes, args.samples_per_episode)
    if policy is not None:
        rows = run_stats_pass(args.repo_id, args.dataset_root, delta_timestamps, groups, policy, preprocessor, q01, q99, args.device)
        write_csv(args.output_dir / "goal_pose_errors.csv", [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
        save_stat_plots(rows, args.output_dir)
        err3d = np.asarray([r["err3d"] for r in rows])
        print(f"[stats] mean err3d={err3d.mean():.4f}m median={np.median(err3d):.4f}m over {len(rows)} samples", flush=True)

    clip_episode_pool = [ep for ep in manifest["episode_index"].unique().tolist() if ep not in {g[0] for g in groups}]
    clip_episodes = rng.choice(clip_episode_pool, size=min(args.num_clips, len(clip_episode_pool)), replace=False).tolist()
    render_clips(
        args.repo_id, args.dataset_root, delta_timestamps, clip_episodes, policy, preprocessor,
        q01, q99, args.device, args.camera, intrinsics, args.clip_length, args.output_dir,
    )
    print(f"[viz] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
