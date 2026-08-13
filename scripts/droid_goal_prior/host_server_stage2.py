#!/usr/bin/env python3
"""MolmoAct2-DROID inference server for a locally-trained goal-pose-prior
checkpoint (Stage1/Stage2 `lerobot_train` output), NOT the released
`allenai/MolmoAct2-DROID` HF Hub snapshot.

This is NOT a drop-in for `examples/droid/host_server_droid.py`. That server
loads a raw HF Transformers snapshot (`AutoModelForImageTextToText` +
`AutoProcessor`, `norm_stats.json` next to the weights) and calls the
upstream model's own `predict_action()` convenience method. A
`lerobot_train` checkpoint dir (`.../checkpoints/<step>/pretrained_model/`)
is a different artifact entirely: a lerobot `PreTrainedPolicy` save (its own
`config.json` + `model.safetensors`, plus separate
`policy_preprocessor.json`/`policy_postprocessor.json` pipelines with
normalizer stats baked into their own `.safetensors` state files instead of
a `norm_stats.json`). It has no bundled `modeling_molmoact2.py` or
tokenizer/processor files, so pointing `host_server_droid.py --repo-id` at
it will not work. This server loads it the same way this repo's own eval
tooling does (see `viz_goal_pose.py`'s `load_policy_and_preprocessor`, in
`scripts/libero_goal_prior/`): `PreTrainedConfig.from_pretrained` +
`lerobot.policies.factory.make_policy` / `make_pre_post_processors`, then
calls the policy's own `predict_action_chunk(batch,
inference_action_mode="continuous")` -- lerobot's native inference API, not
the HF-snapshot `predict_action()` one.

Two schema differences from `examples/droid/host_server_droid.py` worth
flagging explicitly:

1. THREE cameras, not two. This checkpoint's `image_keys` are
   `exterior_1_left`, `exterior_2_left`, `wrist_left` (DROID's two external
   ZED views + wrist), vs. the released checkpoint's `external_cam`/
   `wrist_cam`. To stay a drop-in for existing 2-camera clients --
   `sim_eval`'s `DroidClient` (ManiSkill's `FrankaDROID` agent only ever
   renders `external_cam`/`wrist_cam`, see `sim_eval/robots/franka_droid.py`)
   and `logs/inference_script.py` (the real-robot bridge) -- this server
   ALSO accepts that 2-key payload and duplicates `external_cam` into both
   exterior slots (see `_resolve_images`). That is a real approximation, not
   a free lunch: the checkpoint was trained on two genuinely different DROID
   stereo viewpoints, and both sim and the current robot bridge only ever
   have one exterior view to give it. A warning is logged on first use.
   Prefer sending real `exterior_1_left`/`exterior_2_left`/`wrist_left`
   frames whenever more than one exterior camera is actually available.
2. No goal-pose input needed at inference. Training packed a real future
   `observation.ee_pose` as an auxiliary reconstruction target
   (`enable_pose_reconstruction`), but this checkpoint's
   `goal_token_source="learnable_queries"` means the action expert's goal
   conditioning comes from learned queries, not the packed goal pose --
   `MolmoAct2PackInputsProcessorStep._extract_goal_pose` returns `(None,
   None)` whenever the goal feature lacks the multi-frame time axis dataset
   episodes carry (see its docstring), which is exactly what a live
   single-frame request looks like. So we never send `observation.ee_pose`
   here; omitting it is the *correct* inference-time behavior, not a
   shortcut.

Also unlike the released servers, loading requires read access to the
training dataset's `meta/` directory (`--dataset-root`, default
`/scratch/shailesh.xml/datasets/droid_1.0.1_goal_pose`) -- `make_policy`
needs a `LeRobotDatasetMetadata` to resolve `output_features` (action
feature names/shapes), and `make_pre_post_processors` wants
`dataset_stats` as the base the checkpoint's saved normalizer state loads
on top of. Only `meta/*.json`/`meta/episodes/*.parquet` are read, not the
video/data shards, but it means this server can only run somewhere with
`/scratch` mounted (a Hopper compute node), not on the NUC next to the
robot the way the released servers are meant to.

No bf16 hand-patching needed here (contrast with `host_server_droid.py`'s
`_patch_modeling_for_bf16`): the checkpoint's weights are natively bf16
on disk (confirmed 1579/1580 tensors `BF16`), and lerobot's own
`predict_action_chunk` already wraps the forward pass in
`torch.autocast(dtype=bfloat16)`, so fp32 pixel_values from the processor
get cast automatically -- the manual `_move_and_cast` dtype patch that the
raw-HF-snapshot path needs has no equivalent problem to solve here.

Wire protocol (mirrors host_server_droid.py's shape, different keys):

    GET  /act        -> health check
    POST /act        -> action inference
        request body (json_numpy):
            {
              "exterior_1_left": ndarray(H, W, 3) uint8 RGB,
              "exterior_2_left": ndarray(H, W, 3) uint8 RGB,
              "wrist_left":      ndarray(H, W, 3) uint8 RGB,
              "instruction":     str,
              "state":           ndarray(8,) float32  [q1..q7, gripper],
              "num_steps":       int (optional, flow-matching denoise steps),
              "timestamp":       float (optional),
            }
        response body (json_numpy):
            {"actions": ndarray(chunk_size, 8) float32, "dt_ms": float}

Run (must use the lerobot submodule's venv, not the top-level repo's):

    lerobot/.venv/bin/python scripts/droid_goal_prior/host_server_stage2.py \\
        --checkpoint /scratch/shailesh.xml/outputs/droid_goal_prior/seed_1000/stage2_omp/checkpoints/010000/pretrained_model \\
        --host 0.0.0.0 --port 8100
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import json_numpy
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

# NOTE: FastAPI/Request must be imported at MODULE level, not inside
# build_app(). With `from __future__ import annotations` (below... actually
# above this point) every annotation becomes a lazy string, so
# `async def act(request: Request)` is stored as the literal string
# "Request". FastAPI resolves that string via the function's __globals__ --
# if `Request` were only imported inside build_app() (a nested scope, as an
# earlier version of this file did), it wouldn't be in __globals__ and the
# lookup silently fails; FastAPI then falls back to treating `request` as an
# ordinary field to validate, sourced from the query string, which is always
# empty -> every POST /act fails with 422 "Field required" for
# loc=["query","request"] before the handler body ever runs. Caught via a
# local fastapi.testclient.TestClient repro against a fake Policy, not by
# reading the traceback alone -- the error message doesn't mention imports at
# all, it looks like a client-payload problem.

# NOTE: do NOT call json_numpy.patch() here. It monkeypatches the stdlib
# `json` module process-wide, and `Policy.__init__` below lazily imports
# `lerobot` -> `transformers` -> `scipy` -> `numpy.testing`, which does its
# own unrelated `json.loads()` calls at import time. json_numpy's patched
# decoder hook assumes every decoded object is dict-like and crashes
# (`TypeError: argument of type 'types.SimpleNamespace' is not iterable`) on
# those. `examples/droid/host_server_droid.py` avoids this the same way:
# import the heavy stack first, patch json only once it's done. See main().

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("molmoact2.stage2_server")

WS = Path(__file__).resolve().parents[2]
IMAGE_KEYS = ("exterior_1_left", "exterior_2_left", "wrist_left")
STATE_DIM = 8
DEFAULT_DATASET_ROOT = Path("/scratch/shailesh.xml/datasets/droid_1.0.1_goal_pose")
DEFAULT_REPO_ID = "lerobot/droid_1.0.1"


def _ensure_lerobot_on_path() -> None:
    src = WS / "lerobot" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


class Policy:
    """Loads a lerobot-native MolmoAct2 checkpoint + its pre/post-processor
    pipelines, and serializes inference calls (action-expert generation is
    not safe under concurrent requests -- same reasoning as the released
    servers' lock)."""

    def __init__(self, checkpoint: Path, device: str, dataset_root: Path, repo_id: str) -> None:
        _ensure_lerobot_on_path()
        import lerobot.policies.factory  # noqa: F401  registers "molmoact2" with PreTrainedConfig
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        log.info("Loading checkpoint config from %s", checkpoint)
        cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
        cfg.pretrained_path = str(checkpoint)
        cfg.device = device

        log.info("Loading dataset metadata (meta/ only) from %s", dataset_root)
        ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=str(dataset_root))

        log.info("Building policy (this loads model.safetensors)")
        self.policy = make_policy(cfg, ds_meta=ds_meta)
        self.policy.to(device)
        self.policy.eval()
        if getattr(self.policy.config, "inference_action_mode", None) in (None, ""):
            self.policy.config.inference_action_mode = "continuous"

        log.info("Building pre/post-processor pipelines")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg, pretrained_path=str(checkpoint), dataset_stats=ds_meta.stats
        )

        self.device = device
        self.chunk_size = int(self.policy.config.n_action_steps)
        self.state_dim = STATE_DIM
        self._lock = threading.Lock()

    def predict(
        self,
        images: dict[str, np.ndarray],
        instruction: str,
        state: np.ndarray,
        num_steps: int | None = None,
    ) -> np.ndarray:
        import torch

        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_f32.shape != (self.state_dim,):
            raise ValueError(f"state must be shape ({self.state_dim},), got {state_f32.shape}")

        item: dict[str, Any] = {"observation.state": state_f32, "task": [instruction]}
        for key in IMAGE_KEYS:
            if key not in images:
                raise ValueError(f"missing required camera frame: {key!r}")
            item[f"observation.images.{key}"] = _to_hwc_uint8(images[key])

        with self._lock, torch.inference_mode():
            processed = self.preprocessor(item)
            chunk = self.policy.predict_action_chunk(
                processed, inference_action_mode="continuous", num_steps=num_steps
            )
            actions = self.postprocessor(chunk)

        if torch.is_tensor(actions):
            actions = actions.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return actions[:, : self.state_dim]


def _to_hwc_uint8(arr: Any) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


# Wire schema `sim_eval`'s DroidClient and `logs/inference_script.py` already
# speak (RemoteServerSchema "droid" in sim_eval/inference/common.py) -- one
# exterior view, not two. Accepted as a fallback so both can hit this server
# unmodified; see the module docstring for the duplication caveat.
LEGACY_DROID_CAMERA_KEYS = ("external_cam", "wrist_cam")
_warned_legacy_cameras = False


def _resolve_images(payload: dict) -> dict[str, np.ndarray]:
    global _warned_legacy_cameras
    if all(key in payload for key in IMAGE_KEYS):
        return {key: payload[key] for key in IMAGE_KEYS}
    if all(key in payload for key in LEGACY_DROID_CAMERA_KEYS):
        if not _warned_legacy_cameras:
            log.warning(
                "Received legacy 2-camera payload (%s) -- this checkpoint wants two "
                "DISTINCT exterior views plus a wrist view. Duplicating external_cam "
                "into both exterior_1_left/exterior_2_left; this is an approximation "
                "of what the checkpoint was trained on, not equivalent to it. "
                "(logged once)",
                LEGACY_DROID_CAMERA_KEYS,
            )
            _warned_legacy_cameras = True
        ext = payload["external_cam"]
        return {"exterior_1_left": ext, "exterior_2_left": ext, "wrist_left": payload["wrist_cam"]}
    missing = [key for key in IMAGE_KEYS if key not in payload]
    raise ValueError(
        f"payload is missing camera fields: need either {IMAGE_KEYS} or "
        f"{LEGACY_DROID_CAMERA_KEYS}; missing {missing}"
    )


def build_app(policy: Policy, checkpoint: Path):
    app = FastAPI(title="MolmoAct2-DROID Stage2 checkpoint server", version="0.1.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "checkpoint": str(checkpoint),
                "camera_keys": list(IMAGE_KEYS),
                "legacy_camera_keys_accepted": list(LEGACY_DROID_CAMERA_KEYS),
                "state_dim": policy.state_dim,
                "chunk_size": policy.chunk_size,
                "device": policy.device,
                "dtype": str(next(policy.policy.parameters()).dtype),
            }
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/act")
    async def act(request: Request) -> Response:
        raw = await request.body()
        try:
            payload = json_numpy.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            return _error_response(400, f"failed to decode json_numpy body: {e}")

        try:
            images = _resolve_images(payload)
            instruction = str(payload["instruction"])
            state = payload["state"]
        except KeyError as e:
            return _error_response(400, f"missing required field: {e}")
        except ValueError as e:
            return _error_response(400, str(e))

        num_steps = payload.get("num_steps")
        num_steps = int(num_steps) if num_steps is not None else None

        t0 = time.perf_counter()
        try:
            actions = policy.predict(images=images, instruction=instruction, state=state, num_steps=num_steps)
        except Exception as e:  # noqa: BLE001
            log.exception("inference failed")
            return _error_response(500, f"inference failed: {e}")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        body = json_numpy.dumps({"actions": actions, "dt_ms": dt_ms})
        return Response(content=body, media_type="application/json")

    return app


def _error_response(status: int, message: str) -> Response:
    body = json_numpy.dumps({"error": message})
    return Response(content=body, status_code=status, media_type="application/json")


def warmup(policy: Policy) -> None:
    log.info("Warming up model with a dummy frame ...")
    dummy_img = np.zeros((180, 320, 3), dtype=np.uint8)
    dummy_state = np.zeros(policy.state_dim, dtype=np.float32)
    t0 = time.perf_counter()
    try:
        policy.predict(
            images={key: dummy_img for key in IMAGE_KEYS},
            instruction="warmup",
            state=dummy_state,
        )
    except Exception:  # noqa: BLE001
        log.exception("warmup inference failed (server will still start)")
        return
    log.info("Warmup OK (%.1f ms)", (time.perf_counter() - t0) * 1000.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MolmoAct2-DROID Stage2-checkpoint inference server")
    p.add_argument("--checkpoint", type=Path, required=True, help="path to a .../checkpoints/<step>/pretrained_model dir")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-warmup", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint
    if checkpoint.name != "pretrained_model" and (checkpoint / "pretrained_model").is_dir():
        log.info("Given a checkpoint step dir, using its pretrained_model/ subdir")
        checkpoint = checkpoint / "pretrained_model"

    policy = Policy(checkpoint=checkpoint, device=args.device, dataset_root=args.dataset_root, repo_id=args.repo_id)

    # Safe to patch now: the lerobot/transformers/scipy import chain pulled in
    # by Policy.__init__ is done, so json_numpy's monkeypatch of the stdlib
    # `json` module can no longer collide with an unrelated json.loads() call
    # buried in one of those imports (see the module docstring note at top).
    json_numpy.patch()

    if not args.no_warmup:
        warmup(policy)

    app = build_app(policy, checkpoint)

    import uvicorn

    log.info("Listening on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
