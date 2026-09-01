#!/usr/bin/env python3
"""Bimanual-YAM inference server for a locally-trained goal-pose-prior
checkpoint (Stage1/Stage2 `lerobot_train` output), NOT the released
`allenai/MolmoAct2-BimanualYAM` HF Hub snapshot.

===============================================================================
WHY THIS IS NOT `examples/yam/host_server_yam.py`
===============================================================================

`examples/yam/host_server_yam.py` serves the RELEASED checkpoint. It is a
different artifact with a different contract, and the two are NOT
interchangeable in either direction:

    | | released server | THIS server |
    | --- | --- | --- |
    | artifact | HF snapshot (`AutoModelForImageTextToText`) | lerobot `PreTrainedPolicy` save |
    | normalization | `norm_stats.json` + `norm_tag="yam_dual_molmoact2"` | baked into the checkpoint's own `policy_preprocessor_*.safetensors` |
    | state | **14-D joint angles** (7 per arm) | **16-D absolute EEF pose** (8 per arm) |
    | action | 14-D joint | **16-D absolute EEF pose** |
    | inference call | upstream `predict_action()` | lerobot `predict_action_chunk()` |
    | camera keys | `top_cam` / `left_cam` / `right_cam` | `top` / `left` / `right` |

Pointing `host_server_yam.py --repo-id` at a `lerobot_train` checkpoint dir
will not work: that dir has no bundled `modeling_molmoact2.py`, no
tokenizer/processor files, and no `norm_stats.json`. It carries its own
`config.json` + `model.safetensors` plus separate
`policy_preprocessor.json` / `policy_postprocessor.json` pipelines whose
normalizer statistics live in their own `.safetensors` state files. This
server loads it the way the repo's own training/eval tooling does:
`PreTrainedConfig.from_pretrained` + `lerobot.policies.factory.make_policy`
/ `make_pre_post_processors`.

Structurally this file is a port of
`scripts/droid_goal_prior/host_server_stage2.py`. Read that one too if you
are changing the loading path; the differences are embodiment-specific and
are all called out below.

===============================================================================
THE BIG ONE: ABSOLUTE END-EFFECTOR SPACE
===============================================================================

This is the single most important thing for a client author to get right,
and it is where YAM departs from every other MolmoAct2 deployment in this
repo.

LIBERO and the released DROID/YAM checkpoints emit **delta** actions -- the
robot controller integrates them relative to its current pose. This
checkpoint does **NOT**. It emits **ABSOLUTE end-effector poses in the robot
base frame**, in exactly the same 16-D layout as `observation.state`:

    index:   0    1    2     3    4    5    6       7
            [x,   y,   z,   qw,  qx,  qy,  qz,  gripper]   <- LEFT arm
    index:   8    9   10    11   12   13   14      15
            [x,   y,   z,   qw,  qx,  qy,  qz,  gripper]   <- RIGHT arm

So a returned action row is a *target pose to move to*, not an increment to
add. A client that integrates these (`pose += action`) will drive the arms
away immediately and violently. If you are porting a LIBERO or DROID bridge,
this is the line you must delete.

Consequences a client MUST handle:

1. **DO NOT INTEGRATE.** Command each row as an absolute Cartesian target.
   Your IK / Cartesian controller consumes it directly.

2. **RENORMALIZE THE QUATERNIONS.** Indices 3:7 (left) and 11:15 (right) are
   unit quaternions in **w-first** order `(qw, qx, qy, qz)`. The flow-matching
   head regresses all 16 components independently under per-component q01/q99
   normalization -- nothing couples the four quaternion columns, and nothing
   in the denormalization restores unit norm. A returned quaternion will be
   *close to* unit (e.g. 0.997) but not exactly unit, which is not a valid
   rotation. This server renormalizes them for you by default (see
   `--no-quat-normalize` to disable and inspect the raw head output, and the
   `quat_norm_dev` field in the response). Every training quaternion was
   exactly unit -- `prepare_dataset.py::_check_quaternions` enforces
   `|q| = 1` to within 1e-4 -- so a non-unit quaternion is purely a
   model-output artifact, never something the data taught it.

   Note on **double cover**: `q` and `-q` are the same rotation.
   `prepare_dataset.py::QuaternionCanonicalizer` aligned every episode into a
   single hemisphere before training (fixing 374 raw sign flips), and the
   published data sits comfortably at `qy > 0.59` throughout, so the model
   should not emit hemisphere-flipped output. This server does not force a
   hemisphere; if your controller is sign-sensitive, dot the result against
   your current orientation and negate on a negative dot.

3. **GRIPPER DIMS ARE NOT NORMALIZED.** Indices 7 and 15 are passed through
   raw by the normalizer (its `mask` tensor reads
   `[1,1,1,1,1,1,1,0, 1,1,1,1,1,1,1,0]` -- the two zeros are the grippers).
   They already live in a natural ~[0, 1] range, so normalizing would have
   been redundant and would have distorted the open/close semantics. Do not
   apply any additional scaling to them.

4. **CHUNK SEMANTICS.** The response contains `chunk_size` (30) rows =
   1.0 s of motion at the 30 fps the data was recorded at. Row `i` is the
   intended absolute pose at `t + i + 1`. Execute them in order at ~30 Hz,
   or re-query more often and execute a prefix (receding horizon); both are
   valid, the latter is more robust to model error.

===============================================================================
NO GOAL POSE IS SENT AT INFERENCE (and that is correct)
===============================================================================

Training supervised an auxiliary reconstruction of the FUTURE state
`observation.state` at `t + 30` (`enable_pose_reconstruction=true`,
`target_pose_delta_index=30`, weight 0.3). That is a TRAINING SIGNAL ONLY.

At inference the future pose is unknowable -- that is the entire premise of
the method. Stage 2's `goal_token_source="learnable_queries"` +
`goal_conditioning_mode="semantic_visual_recurrent"` means the action
expert's conditioning is aggregated from 100 learned latent tokens that read
image/language/state context, not from any packed goal pose. Concretely,
`MolmoAct2PackInputsProcessorStep._extract_goal_pose` returns `(None, None)`
whenever the goal feature lacks the multi-frame time axis that dataset
episodes carry -- which is exactly what a live single-frame request looks
like. So this server never sends `observation.state` with a time axis, and
omitting the goal is the *correct* inference-time behavior rather than an
approximation.

===============================================================================
CAMERAS
===============================================================================

Three streams, and **the key names and their meaning both matter** -- they
must match what training packed, or the model silently attends to the wrong
viewpoint:

    top    -> `observation.images.top`     (360 x 640 in training)
    left   -> `observation.images.left`    (480 x 640 in training)
    right  -> `observation.images.right`   (480 x 640 in training)

The processor resizes internally, so exact incoming resolution is not
critical, but ASPECT RATIO and VIEWPOINT ASSIGNMENT are: sending the right
wrist camera under the `left` key will not error, it will just quietly
degrade the policy. There is deliberately NO legacy/aliasing fallback here
(unlike the DROID server, which duplicates one exterior view into two slots
for 2-camera clients) -- YAM has three genuinely distinct cameras and every
YAM client has all three, so silently guessing would only hide bugs.

Images are `HxWx3` **uint8 RGB**. If your camera stack hands you BGR (OpenCV
does), convert before sending.

===============================================================================
WIRE PROTOCOL
===============================================================================

Both directions are `json_numpy`-encoded, so ndarrays round-trip natively.

    GET  /act       -> health/introspection (also GET /healthz for liveness)

    POST /act       -> action inference

      request body:
        {
          "top":         ndarray(H, W, 3) uint8 RGB,   # required
          "left":        ndarray(H, W, 3) uint8 RGB,   # required
          "right":       ndarray(H, W, 3) uint8 RGB,   # required
          "instruction": str,                          # required, natural language
          "state":       ndarray(16,) float32,         # required, ABSOLUTE EEF (see layout above)
          "num_steps":   int,                          # optional, flow denoise steps
          "timestamp":   float,                        # optional, echoed back, for client logging
        }

      response body:
        {
          "actions":       ndarray(30, 16) float32,  # ABSOLUTE EEF poses, NOT deltas
          "dt_ms":         float,                    # server-side inference latency
          "quat_norm_dev": float,                    # max |1 - |q|| BEFORE renormalization
          "timestamp":     float,                    # echoed from request if provided
        }

`quat_norm_dev` is a free health metric: it should be small (<~0.05). If it
climbs over a session, the pose head is drifting off the rotation manifold
and the checkpoint is suspect -- worth logging on the client side.

Minimal client:

    import json_numpy, requests, numpy as np
    json_numpy.patch()

    resp = requests.post(
        "http://<host>:8203/act",
        json={
            "top":   top_rgb_uint8,      # (360, 640, 3)
            "left":  left_rgb_uint8,     # (480, 640, 3)
            "right": right_rgb_uint8,    # (480, 640, 3)
            "instruction": "Put all blocks into the box.",
            "state": state16_float32,    # ABSOLUTE EEF, w-first quats
        },
        timeout=30,
    ).json()

    actions = resp["actions"]            # (30, 16) ABSOLUTE poses
    for target in actions:               # execute at ~30 Hz
        robot.command_absolute_eef(target)   # NOT robot.pose += target

The three training instructions (use these verbatim for in-distribution
behavior; the model saw only these three):

    "Put all blocks into the box."
    "Clean the table using the dust pan."
    "Transfer the egg from the pan into the bowl."

===============================================================================
OPERATIONAL NOTES
===============================================================================

* Must run with the lerobot submodule's venv (`lerobot/.venv/bin/python`),
  not the top-level repo venv.
* Loading needs read access to the training dataset's `meta/` directory
  (`--dataset-root`): `make_policy` needs `LeRobotDatasetMetadata` to resolve
  `output_features`, and `make_pre_post_processors` wants `dataset_stats` as
  the base the checkpoint's saved normalizer state loads on top of. Only
  `meta/*.json` and `meta/episodes/*.parquet` are read -- never the video or
  data shards -- but it does mean this server needs the dataset directory
  present, so it cannot run on an isolated robot-side NUC the way the
  released servers can.
* No bf16 hand-patching (contrast `examples/yam/host_server_yam.py`'s
  `_patch_modeling_for_bf16`): checkpoint weights are natively bf16 and
  lerobot's `predict_action_chunk` wraps the forward in
  `torch.autocast(bfloat16)`, so the fp32 `pixel_values` the processor
  produces get cast automatically.
* Inference is serialized under a lock. The action expert's generation path
  is not safe under concurrent calls -- same reasoning as the released
  servers.

Run:

    lerobot/.venv/bin/python scripts/yam_goal_prior/host_server_stage2.py \\
        --checkpoint lerobot/outputs/yam_goal_prior/seed_1000/stage2/checkpoints/last/pretrained_model \\
        --host 0.0.0.0 --port 8203
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

# NOTE: FastAPI / Request must be imported at MODULE level, not inside
# build_app(). With `from __future__ import annotations` every annotation
# becomes a lazy string, so `async def act(request: Request)` is stored as
# the literal string "Request". FastAPI resolves that string via the
# function's __globals__ -- if `Request` were imported only inside
# build_app() (a nested scope) the lookup silently fails, FastAPI falls back
# to treating `request` as an ordinary query-string field to validate, and
# every POST /act then fails with 422 "Field required" for
# loc=["query","request"] before the handler body ever runs. The error message
# does not mention imports at all. Inherited verbatim from the DROID server,
# where it was diagnosed the hard way.

# NOTE: do NOT call json_numpy.patch() at import time. It monkeypatches the
# stdlib `json` module process-wide, and `Policy.__init__` lazily imports
# lerobot -> transformers -> scipy -> numpy.testing, which makes its own
# unrelated json.loads() calls at import time. json_numpy's patched decoder
# hook assumes every decoded object is dict-like and crashes on those. Patch
# only after the heavy imports are done -- see main().

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("molmoact2.yam_stage2_server")

WS = Path(__file__).resolve().parents[2]

# Camera keys, in the order training packed them. These become
# `observation.images.<key>`; see the checkpoint's `image_keys`.
IMAGE_KEYS = ("top", "left", "right")

# 16-D absolute EEF pose: [xyz(3), quat_wxyz(4), gripper(1)] x 2 arms.
STATE_DIM = 16

# Slices of the quaternion blocks within the 16-D vector, per arm. Used for
# unit-norm restoration on the model's output. Left arm occupies 0:8, right
# arm 8:16; within each, the quaternion is at local offset 3:7.
ARM_OFFSETS = (0, 8)
QUAT_SLICES = tuple(slice(off + 3, off + 7) for off in ARM_OFFSETS)

# Gripper indices -- listed for documentation/assertion purposes. These are
# the two dims the normalizer leaves raw (its mask is 0 at these positions),
# and the two that must NOT be touched by any client-side rescaling.
GRIPPER_INDICES = (7, 15)

DEFAULT_DATASET_ROOT = WS / "real_robot_datasets" / "yam_3task_goal_pose"
DEFAULT_REPO_ID = "local/yam_blocks_goal_pose"
DEFAULT_PORT = 8203  # released YAM server owns 8202; DROID stage2 owns 8100

# The exact instruction strings the checkpoint was trained on. Anything else
# is out of distribution -- not an error, but worth warning about once.
TRAINING_INSTRUCTIONS = (
    "Put all blocks into the box.",
    "Clean the table using the dust pan.",
    "Transfer the egg from the pan into the bowl.",
)


def _ensure_lerobot_on_path() -> None:
    """Prepend the lerobot submodule's `src/` so `import lerobot` resolves to
    this checkout rather than any site-packages copy."""
    src = WS / "lerobot" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _to_hwc_uint8(arr: Any) -> np.ndarray:
    """Coerce an incoming frame to HxWx3 uint8, erroring loudly on anything
    that is not plausibly an RGB image. Note we cannot detect BGR-vs-RGB --
    that is on the client."""
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {a.shape}")
    if a.dtype != np.uint8:
        # Float images in [0,1] would clip to all-zeros here, which would be a
        # silent disaster, so reject them explicitly rather than guessing.
        if np.issubdtype(a.dtype, np.floating) and float(np.nanmax(a)) <= 1.0 + 1e-6:
            raise ValueError(
                "image looks like float in [0,1]; send uint8 RGB in [0,255] "
                "(multiply by 255 and cast before sending)"
            )
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def normalize_quaternions(actions: np.ndarray) -> tuple[np.ndarray, float]:
    """Restore unit norm on both arms' quaternion blocks.

    The flow-matching head regresses all 16 action components independently
    under per-component q01/q99 normalization. Nothing in the model or in the
    denormalization couples the four quaternion columns, so the emitted
    quaternion is close to -- but not exactly -- unit norm. A non-unit
    quaternion is not a rotation: downstream rotation math either silently
    produces a slightly wrong orientation or raises, depending on the library.

    Returns `(fixed_actions, max_norm_deviation)` where the deviation is
    `max |1 - |q||` measured BEFORE the fix, across every row and both arms.
    That number is a useful health signal, so it is surfaced in the response.

    Positions (0:3, 8:11) and grippers (7, 15) are untouched -- rescaling
    those would corrupt real measurements.
    """
    fixed = np.array(actions, dtype=np.float32, copy=True)
    worst = 0.0
    for quat in QUAT_SLICES:
        block = fixed[:, quat]                       # (chunk, 4)
        norms = np.linalg.norm(block, axis=1)        # (chunk,)
        worst = max(worst, float(np.max(np.abs(1.0 - norms))))
        # Guard against a pathological all-zero row rather than dividing by 0.
        safe = np.where(norms > 1e-8, norms, 1.0)
        fixed[:, quat] = block / safe[:, None]
    return fixed, worst


class Policy:
    """Loads a lerobot-native MolmoAct2 checkpoint plus its pre/post-processor
    pipelines, and serializes inference.

    The lock is not optional: the action expert's generation path is not safe
    under concurrent calls, and a robot client polling at ~30 Hz will happily
    overlap requests if you let it.
    """

    def __init__(
        self,
        checkpoint: Path,
        device: str,
        dataset_root: Path,
        repo_id: str,
        quat_normalize: bool = True,
    ) -> None:
        _ensure_lerobot_on_path()
        import lerobot.policies.factory  # noqa: F401  registers "molmoact2" with PreTrainedConfig
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        log.info("Loading checkpoint config from %s", checkpoint)
        cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
        cfg.pretrained_path = str(checkpoint)
        cfg.device = device

        # meta/ only -- no video or data shards are read. See module docstring.
        log.info("Loading dataset metadata (meta/ only) from %s", dataset_root)
        ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=str(dataset_root))

        log.info("Building policy (this loads model.safetensors, ~11 GB)")
        self.policy = make_policy(cfg, ds_meta=ds_meta)
        self.policy.to(device)
        self.policy.eval()

        # Stage-2 checkpoints are saved with `inference_action_mode` unset
        # (it is a runtime choice, not a trained property). The goal-pose
        # config validation requires continuous mode, and the policy-owned
        # prefill/denoise path -- which is what actually routes through the
        # learned queries, the synthetic KV, and
        # `mask_image_from_action_expert` -- is only taken in continuous mode.
        # Leaving this unset would fall back to a path that silently bypasses
        # all three. See the v2b implementation notes on
        # `_uses_policy_continuous_generation`.
        if getattr(self.policy.config, "inference_action_mode", None) in (None, ""):
            self.policy.config.inference_action_mode = "continuous"

        log.info("Building pre/post-processor pipelines")
        # `dataset_stats` is the base; the checkpoint's own saved normalizer
        # state (policy_preprocessor_*.safetensors) loads on top of it. Both
        # were verified identical to meta/stats.json at training time.
        #
        # The `device_processor` override is REQUIRED, not cosmetic. The
        # checkpoint's saved `policy_preprocessor.json` bakes in whatever
        # device training ran on (`"device": "cuda"`), and
        # `PolicyProcessorPipeline.from_pretrained` instantiates that step from
        # the saved config -- it does NOT consult `cfg.device`. Without this
        # override the pipeline ignores `--device` entirely: on a CPU-only host
        # it dies with a bare `AssertionError` from
        # `assert torch.cuda.is_available()` wrapped in an opaque
        # "Failed to instantiate processor step 'device_processor'", and on a
        # multi-GPU host `--device cuda:1` would silently still place tensors
        # via the saved `"cuda"`. Same pattern as
        # `scripts/droid_goal_prior/validate_dataloader.py`.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=str(checkpoint),
            dataset_stats=ds_meta.stats,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        self.device = device
        self.chunk_size = int(self.policy.config.n_action_steps)
        self.state_dim = STATE_DIM
        self.quat_normalize = quat_normalize
        self._lock = threading.Lock()
        self._warned_ood_instruction = False

        # Fail fast on an embodiment mismatch rather than at first inference.
        cfg_keys = [k.rsplit(".", 1)[-1] for k in (getattr(cfg, "image_keys", None) or [])]
        if cfg_keys and tuple(cfg_keys) != IMAGE_KEYS:
            raise ValueError(
                f"checkpoint image_keys {tuple(cfg_keys)} != this server's {IMAGE_KEYS}; "
                "wrong checkpoint for this server?"
            )
        log.info(
            "Ready: chunk_size=%d state_dim=%d cameras=%s quat_normalize=%s",
            self.chunk_size, self.state_dim, IMAGE_KEYS, self.quat_normalize,
        )

    def predict(
        self,
        images: dict[str, np.ndarray],
        instruction: str,
        state: np.ndarray,
        num_steps: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run one inference. Returns `(actions, quat_norm_dev)`.

        `actions` is (chunk_size, 16) float32 of ABSOLUTE EEF poses.
        """
        import torch

        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_f32.shape != (self.state_dim,):
            raise ValueError(
                f"state must be shape ({self.state_dim},) -- 16-D absolute EEF "
                f"[xyz,quat_wxyz,gripper] x 2 arms -- got {state_f32.shape}. "
                "(The RELEASED YAM checkpoint takes 14-D joint angles; this is "
                "a different model.)"
            )

        if instruction not in TRAINING_INSTRUCTIONS and not self._warned_ood_instruction:
            log.warning(
                "Instruction %r is not one of the three the checkpoint was trained on "
                "%s -- behavior is out of distribution. (logged once)",
                instruction, TRAINING_INSTRUCTIONS,
            )
            self._warned_ood_instruction = True

        # Single-frame item, no time axis on observation.state. That is what
        # makes `_extract_goal_pose` return (None, None), which is the correct
        # inference-time behavior for learnable-query conditioning -- see the
        # module docstring.
        item: dict[str, Any] = {"observation.state": state_f32, "task": [instruction]}
        for key in IMAGE_KEYS:
            if key not in images:
                raise ValueError(f"missing required camera frame: {key!r} (need all of {IMAGE_KEYS})")
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
        # The action expert pads to expected_max_action_dim (32); trim to the
        # 16 real dims this embodiment uses.
        actions = actions[:, : self.state_dim]

        if self.quat_normalize:
            actions, quat_dev = normalize_quaternions(actions)
        else:
            _, quat_dev = normalize_quaternions(actions)  # measure only
        return actions, quat_dev


def build_app(policy: Policy, checkpoint: Path):
    app = FastAPI(title="MolmoAct2 Bimanual-YAM goal-pose-prior Stage2 server", version="0.1.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        """Introspection endpoint -- deliberately verbose so a client author
        can verify the contract without reading this file."""
        return JSONResponse(
            {
                "status": "ok",
                "checkpoint": str(checkpoint),
                "camera_keys": list(IMAGE_KEYS),
                "state_dim": policy.state_dim,
                "action_dim": policy.state_dim,
                "action_space": "ABSOLUTE end-effector pose (NOT delta)",
                "action_layout": "[x,y,z,qw,qx,qy,qz,gripper] x 2 arms (left 0:8, right 8:16)",
                "quaternion_order": "w-first (qw,qx,qy,qz)",
                "quaternion_indices": [[3, 7], [11, 15]],
                "gripper_indices": list(GRIPPER_INDICES),
                "gripper_normalized": False,
                "quat_normalize_applied": policy.quat_normalize,
                "chunk_size": policy.chunk_size,
                "control_hz": 30,
                "training_instructions": list(TRAINING_INSTRUCTIONS),
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
            missing = [k for k in IMAGE_KEYS if k not in payload]
            if missing:
                raise ValueError(
                    f"payload is missing camera fields {missing}; need all of {IMAGE_KEYS}. "
                    "(No aliasing fallback exists: YAM has three distinct cameras and "
                    "guessing would hide client bugs.)"
                )
            images = {key: payload[key] for key in IMAGE_KEYS}
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
            actions, quat_dev = policy.predict(
                images=images, instruction=instruction, state=state, num_steps=num_steps
            )
        except ValueError as e:
            # Client-side contract violations (bad shapes/dtypes) are 400s, not 500s.
            return _error_response(400, str(e))
        except Exception as e:  # noqa: BLE001
            log.exception("inference failed")
            return _error_response(500, f"inference failed: {e}")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        out: dict[str, Any] = {
            "actions": actions,
            "dt_ms": dt_ms,
            "quat_norm_dev": quat_dev,
        }
        if "timestamp" in payload:
            out["timestamp"] = payload["timestamp"]
        return Response(content=json_numpy.dumps(out), media_type="application/json")

    return app


def _error_response(status: int, message: str) -> Response:
    body = json_numpy.dumps({"error": message})
    return Response(content=body, status_code=status, media_type="application/json")


def warmup(policy: Policy) -> None:
    """One dummy inference so the first real request is not paying CUDA graph
    / kernel autotune costs. Uses each camera's true training resolution."""
    log.info("Warming up model with dummy frames ...")
    dummy = {
        "top": np.zeros((360, 640, 3), dtype=np.uint8),
        "left": np.zeros((480, 640, 3), dtype=np.uint8),
        "right": np.zeros((480, 640, 3), dtype=np.uint8),
    }
    dummy_state = np.zeros(policy.state_dim, dtype=np.float32)
    # Identity quaternions so the warmup state is at least geometrically valid.
    dummy_state[3] = 1.0   # left  qw
    dummy_state[11] = 1.0  # right qw
    t0 = time.perf_counter()
    try:
        actions, quat_dev = policy.predict(
            images=dummy, instruction=TRAINING_INSTRUCTIONS[0], state=dummy_state
        )
    except Exception:  # noqa: BLE001
        log.exception("warmup inference failed (server will still start)")
        return
    log.info(
        "Warmup OK (%.1f ms), actions%s quat_norm_dev=%.2e",
        (time.perf_counter() - t0) * 1000.0, actions.shape, quat_dev,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bimanual-YAM goal-pose-prior Stage2 checkpoint inference server "
                    "(ABSOLUTE EEF action space)"
    )
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="path to a .../checkpoints/<step>/pretrained_model dir "
             "(a step dir also works; the pretrained_model/ subdir is used)",
    )
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                   help="training dataset dir; only meta/ is read")
    p.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument(
        "--no-quat-normalize", action="store_true",
        help="return the head's raw quaternion output without restoring unit norm. "
             "Diagnostic only -- clients then MUST renormalize themselves.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint
    if checkpoint.name != "pretrained_model" and (checkpoint / "pretrained_model").is_dir():
        log.info("Given a checkpoint step dir, using its pretrained_model/ subdir")
        checkpoint = checkpoint / "pretrained_model"

    policy = Policy(
        checkpoint=checkpoint,
        device=args.device,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        quat_normalize=not args.no_quat_normalize,
    )

    # Safe to patch only now: the lerobot/transformers/scipy import chain
    # pulled in by Policy.__init__ is complete, so json_numpy's monkeypatch of
    # the stdlib `json` module can no longer collide with an unrelated
    # json.loads() buried in one of those imports. See the module-level note.
    json_numpy.patch()

    if not args.no_warmup:
        warmup(policy)

    app = build_app(policy, checkpoint)

    import uvicorn

    log.info("Listening on %s:%d  (POST /act)", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
