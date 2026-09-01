#!/usr/bin/env bash
# Fine-tune MolmoAct2 on the derived bimanual YAM goal-pose dataset.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_yam_molmoact2.sh --sample-manifest PATH [lerobot-train args...]

The manifest must be the one recorded in goal_pose_provenance.json.
SAMPLE_MANIFEST_PATH may be used instead of --sample-manifest.
EOF
}

PASSTHROUGH_ARGS=()
while (($#)); do
  case "$1" in
    --sample-manifest)
      if (($# < 2)); then
        echo "--sample-manifest requires a path" >&2
        exit 2
      fi
      SAMPLE_MANIFEST_PATH="$2"
      shift 2
      ;;
    --sample-manifest=*)
      SAMPLE_MANIFEST_PATH="${1#*=}"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      PASSTHROUGH_ARGS+=("$@")
      break
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

DATASET_REPO_ID="${DATASET_REPO_ID:-local/yam_blocks_goal_pose}"
DATASET_ROOT="${DATASET_ROOT:-${WS}/real_robot_datasets/yam_3task_goal_pose}"

# Checkpoints live at ${WS}/Checkpoint on the DROID hosts but at
# ${HOME}/checkpoints on this one; prefer whichever actually exists.
_resolve_ckpt() {
  local name="$1"
  if [[ -d "${WS}/Checkpoint/${name}" ]]; then
    echo "${WS}/Checkpoint/${name}"
  elif [[ -d "${HOME}/checkpoints/${name}" ]]; then
    echo "${HOME}/checkpoints/${name}"
  else
    echo "${WS}/Checkpoint/${name}"
  fi
}

CHECKPOINT_PATH="${CHECKPOINT_PATH:-$(_resolve_ckpt MolmoAct2)}"
CHECKPOINT_REVISION="${CHECKPOINT_REVISION:-e432d85f6e039edca44afb93c262f3084ab72a9c}"
PROVENANCE_PATH="${PROVENANCE_PATH:-${DATASET_ROOT}/goal_pose_provenance.json}"
SAMPLE_MANIFEST_PATH="${SAMPLE_MANIFEST_PATH:-}"
GOAL_POSE_FEATURE_KEY="${GOAL_POSE_FEATURE_KEY:-observation.state}"
# Optional pin. When set, the build's recorded source content digest must match,
# which is how a locally recorded dataset (no Hugging Face revision) is pinned.
EXPECTED_SOURCE_DIGEST="${EXPECTED_SOURCE_DIGEST:-}"
VALIDATE_DATASET="${VALIDATE_DATASET:-true}"

CHUNK_SIZE="${CHUNK_SIZE:-30}"
EXPECTED_FPS="${EXPECTED_FPS:-30}"
STATE_DIM="${STATE_DIM:-16}"
GOAL_DIM="${GOAL_DIM:-16}"

# State is itself the absolute EEF pose, so the goal is observation.state at
# t+H -- the LIBERO arrangement, with no separate goal feature.
if [[ "${GOAL_POSE_FEATURE_KEY}" != "observation.state" ]]; then
  echo "GOAL_POSE_FEATURE_KEY must be observation.state, got: ${GOAL_POSE_FEATURE_KEY}" >&2
  exit 2
fi
if [[ -z "${SAMPLE_MANIFEST_PATH}" ]]; then
  echo "Missing sample manifest. Pass --sample-manifest PATH." >&2
  exit 2
fi
if [[ "${VALIDATE_DATASET}" != "true" && "${DRY_RUN:-false}" != "true" ]]; then
  echo "VALIDATE_DATASET=false is allowed only with DRY_RUN=true." >&2
  exit 2
fi

if [[ "${VALIDATE_DATASET}" == "true" ]]; then
  "${WS}/lerobot/.venv/bin/python" - \
    "${DATASET_ROOT}" \
    "${PROVENANCE_PATH}" \
    "${SAMPLE_MANIFEST_PATH}" \
    "${GOAL_POSE_FEATURE_KEY}" \
    "${CHUNK_SIZE}" \
    "${EXPECTED_FPS}" \
    "${STATE_DIM}" \
    "${GOAL_DIM}" \
    "${EXPECTED_SOURCE_DIGEST}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
provenance_path = Path(sys.argv[2]).expanduser().resolve()
manifest_path = Path(sys.argv[3]).expanduser().resolve()
goal_key = sys.argv[4]
horizon = int(sys.argv[5])
expected_fps = int(sys.argv[6])
state_dim = int(sys.argv[7])
goal_dim = int(sys.argv[8])
expected_digest = sys.argv[9]
info_path = root / "meta" / "info.json"
stats_path = root / "meta" / "stats.json"

CAMERA_KEYS = (
    "observation.images.top",
    "observation.images.left",
    "observation.images.right",
)


def refuse(message: str) -> None:
    raise SystemExit(f"REFUSING: {message}")


def load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        refuse(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        refuse(f"invalid {label} {path}: {exc}")
    if not isinstance(value, dict):
        refuse(f"{label} must be a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector(value: object, dim: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != dim:
        refuse(f"{label} must have shape [{dim}]")
    if any(
        isinstance(item, (list, dict, bool)) or not isinstance(item, (int, float))
        for item in value
    ):
        refuse(f"{label} must contain {dim} numeric scalars")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        refuse(f"{label} contains a non-finite value")
    return result


provenance = load_json(provenance_path, "provenance")
config = provenance.get("config")
artifacts = provenance.get("artifact_sha256")
if not isinstance(config, dict) or not isinstance(artifacts, dict):
    refuse("provenance must contain config and artifact_sha256 objects")
required_provenance = {
    "format_version": (provenance.get("format_version"), 1),
    "recipe_version": (provenance.get("recipe_version"), 2),
    "source_read_only": (provenance.get("source_read_only"), True),
    "fps": (provenance.get("fps"), expected_fps),
    "config.horizon": (config.get("horizon"), horizon),
    "anchor_definition.goal_offset": (
        provenance.get("anchor_definition", {}).get("goal_offset"),
        horizon,
    ),
    "state_action_definition.dtype": (
        provenance.get("state_action_definition", {}).get("dtype"),
        "float32",
    ),
    "state_action_definition.dim": (
        provenance.get("state_action_definition", {}).get("dim"),
        state_dim,
    ),
    "state_action_definition.rotation_encoding": (
        provenance.get("state_action_definition", {}).get("rotation_encoding"),
        "quaternion_wxyz",
    ),
    "goal_definition.target_feature": (
        provenance.get("goal_definition", {}).get("target_feature"),
        goal_key,
    ),
    "goal_definition.offset": (
        provenance.get("goal_definition", {}).get("offset"),
        horizon,
    ),
}
for label, (actual, expected) in required_provenance.items():
    if actual != expected:
        refuse(f"provenance {label} must be {expected!r}, got {actual!r}")
if Path(str(config.get("output_root", ""))).expanduser().resolve() != root:
    refuse("provenance output_root does not match DATASET_ROOT")
if expected_digest and config.get("source_digest") != expected_digest:
    refuse(
        "provenance source_digest does not match EXPECTED_SOURCE_DIGEST: "
        f"{config.get('source_digest')!r}"
    )
success = load_json(root / "_SUCCESS.json", "completion marker")
if success.get("status") != "complete":
    refuse("completion marker status must be 'complete'")
if success.get("config_fingerprint") != provenance.get("config_fingerprint"):
    refuse("completion marker config fingerprint does not match provenance")
if success.get("provenance_sha256") != sha256(provenance_path):
    refuse("completion marker provenance SHA-256 does not match")

if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
    refuse(f"sample manifest is missing or empty: {manifest_path}")
try:
    manifest_relative = manifest_path.relative_to(root).as_posix()
except ValueError:
    refuse(f"sample manifest must be inside DATASET_ROOT: {manifest_path}")
if manifest_relative != "valid_anchor_indices.parquet":
    refuse(
        "sample manifest must be DATASET_ROOT/valid_anchor_indices.parquet, "
        f"got {manifest_relative}"
    )
manifest_hash = sha256(manifest_path)
if artifacts.get(manifest_relative) != manifest_hash:
    refuse(
        "sample manifest SHA-256 mismatch: "
        f"expected {artifacts.get(manifest_relative)!r}, got {manifest_hash}"
    )

# Data-file integrity. The DROID recipe hashed only metadata here, so 33
# truncated parquet files passed every provenance check and killed the run
# inside the dataloader after it had already claimed the GPUs.
data_files = provenance.get("files")
if not isinstance(data_files, list) or not data_files:
    refuse("provenance contains no data file records")
for record in data_files:
    relative = str(record.get("path", ""))
    path = root / relative
    if not path.is_file():
        refuse(f"missing derived data file: {relative}")
    digest = sha256(path)
    if record.get("output_sha256") != digest:
        refuse(f"derived data file SHA-256 mismatch: {relative}")
    if artifacts.get(relative) != digest:
        refuse(f"data file is not covered by artifact_sha256: {relative}")
print(f"[yam] verified {len(data_files)} derived data file(s) by SHA-256")

info = load_json(info_path, "dataset info")
features = info.get("features")
if not isinstance(features, dict):
    refuse(f"{info_path} has no features object")
canonical_features = dict.fromkeys(("observation.state", "action", goal_key))
for key in canonical_features:
    dim = goal_dim if key == goal_key else state_dim
    feature = features.get(key)
    if not isinstance(feature, dict) or feature.get("shape") != [dim]:
        refuse(f"{info_path} feature {key!r} must have shape [{dim}]")
    if feature.get("dtype") != "float32":
        refuse(f"{info_path} feature {key!r} must use float32")
# LeRobot maps every key beginning with "action" to a policy action feature, so
# a leftover source column here would collide with the canonical one.
stray = sorted(k for k in features if k.startswith("action") and k != "action")
if stray:
    refuse(f"{info_path} still contains colliding action features: {stray}")

# Camera resolutions are not pinned: these recordings are expected to be redone
# at possibly different resolutions. What must hold is that all three named
# slots exist, are video, and are HxWx3.
camera_shapes = {}
for key in CAMERA_KEYS:
    feature = features.get(key)
    if not isinstance(feature, dict):
        refuse(f"{info_path} is missing camera {key!r}")
    if feature.get("dtype") != "video":
        refuse(f"{info_path} feature {key!r} must use video dtype")
    shape = feature.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or shape[2] != 3
        or not all(isinstance(item, int) and item > 0 for item in shape)
    ):
        refuse(f"{info_path} camera {key!r} must have shape [H, W, 3], got {shape!r}")
    camera_shapes[key] = shape

gripper_dims = None
for key in canonical_features:
    names = features[key].get("names")
    dim = goal_dim if key == goal_key else state_dim
    if not isinstance(names, list) or len(names) != dim:
        refuse(f"{info_path} feature {key!r} must carry {dim} per-dimension names")
    found = [index for index, name in enumerate(names) if "gripper" in str(name).lower()]
    if len(found) != 2:
        refuse(
            f"{key} must name exactly two gripper dimensions so the MolmoAct2 "
            f"normalization mask leaves them raw, found {found}"
        )
    gripper_dims = found

if info.get("fps") != expected_fps:
    refuse(f"{info_path} fps must be {expected_fps}, got {info.get('fps')!r}")
goal_info = info.get("goal_pose_prior")
if not isinstance(goal_info, dict):
    refuse(f"{info_path} has no goal_pose_prior object")
if goal_info.get("horizon") != horizon or goal_info.get("target_feature") != goal_key:
    refuse(f"{info_path} goal_pose_prior contract is invalid")
if goal_info.get("anchor_manifest") != manifest_relative:
    refuse(f"{info_path} anchor_manifest does not match the selected manifest")

stats = load_json(stats_path, "dataset stats")
stats_hash = sha256(stats_path)
if artifacts.get("meta/stats.json") != stats_hash:
    refuse(
        f"stats SHA-256 mismatch: expected {artifacts.get('meta/stats.json')!r}, got {stats_hash}"
    )
for key in canonical_features:
    dim = goal_dim if key == goal_key else state_dim
    item = stats.get(key)
    if not isinstance(item, dict):
        refuse(f"{stats_path} is missing {key!r}")
    values = {
        name: vector(item.get(name), dim, f"{key}.{name}")
        for name in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
    }
    for index in range(dim):
        ordered = [values[name][index] for name in ("min", "q01", "q10", "q50", "q90", "q99", "max")]
        if ordered != sorted(ordered):
            refuse(f"{key} statistics are not monotonic at dimension {index}")
        if values["std"][index] <= 0:
            refuse(f"{key}.std must be positive at dimension {index}")

print(f"[yam] verified provenance: {provenance_path}")
print(f"[yam] verified manifest sha256={manifest_hash}")
print(
    f"[yam] verified stats sha256={stats_hash} and shapes "
    f"state={state_dim}/action={state_dim}/goal={goal_dim} (goal key={goal_key})"
)
for key, shape in camera_shapes.items():
    print(f"[yam] camera {key} shape={shape}")
print(f"[yam] goal gripper dims (left raw by the normalizer) = {gripper_dims}")
PY
else
  echo "[yam] DRY_RUN: dataset file validation disabled"
fi

# shellcheck source=/dev/null
source "${WS}/scripts/activate_train_env.sh"

# Refuse a silent CPU fallback. lerobot only *warns* when policy.device=cuda is
# unavailable ("Device 'cuda' is not available. Switching to 'cpu'") and then
# trains anyway -- on this 5.4B model that is ~195 s/step and produces a
# checkpoint that proves nothing about the GPU path. Seen on crane1, whose
# driver reports CUDA 11.6 while this venv ships torch built for CUDA 12.8.
REQUIRE_CUDA="${REQUIRE_CUDA:-true}"
if [[ "${REQUIRE_CUDA}" == "true" && "${DRY_RUN:-false}" != "true" ]]; then
  "${WS}/lerobot/.venv/bin/python" - "${CUDA_VISIBLE_DEVICES:-}" <<'PYCUDA'
import sys
import torch

requested = [item for item in sys.argv[1].split(",") if item.strip()]


def refuse(message: str) -> None:
    raise SystemExit(f"REFUSING: {message}")


if not torch.cuda.is_available():
    detail = ""
    try:
        torch.zeros(1).cuda()
    except Exception as exc:  # noqa: BLE001 - surfacing the driver message is the point
        detail = f" ({exc})"
    refuse(
        f"CUDA is unavailable, so training would silently fall back to CPU{detail}. "
        f"This venv ships torch {torch.__version__} built for CUDA {torch.version.cuda}; "
        "the node's driver must be new enough for that. Set REQUIRE_CUDA=false only "
        "if you deliberately want a CPU run."
    )
visible = torch.cuda.device_count()
if requested and visible != len(requested):
    refuse(
        f"CUDA_VISIBLE_DEVICES requests {len(requested)} GPU(s) but torch sees {visible}"
    )
names = {torch.cuda.get_device_name(i) for i in range(visible)}
free, total = torch.cuda.mem_get_info(0)
print(
    f"[yam] CUDA OK: {visible} x {'/'.join(sorted(names))} "
    f"({total / 1e9:.0f} GB total, {free / 1e9:.0f} GB free on device 0)"
)
PYCUDA
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES_LOCAL="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"

# Multi-node knobs. NUM_MACHINES=1 makes the rank/ip/port flags no-ops.
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-127.0.0.1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
NUM_PROCESSES="$((NUM_PROCESSES_LOCAL * NUM_MACHINES))"

# `accelerate launch` (unlike torchrun) does NOT set OMP_NUM_THREADS, so torch
# defaults to half the machine's cores in EVERY rank. OpenMP busy-waits at
# barriers, so the cores end up pegged spinning instead of feeding the
# dataloader -- on DROID this pinned data_s at ~100s regardless of NUM_WORKERS.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# MolmoAct2's auto-inferred sequence cap allots only 32 tokens to the task
# string (MOLMOACT2_TASK_TOKEN_BUDGET) and assumes 196 tokens per image. With 3
# cameras and a 16-D state the inferred cap is
# 3*196 + 80 + 32 + 16 + 32 = 748 -> rounded to 768. The oversize check in
# processor_molmoact2.py is a hard ValueError with no truncation path, and
# because SEED makes shuffling deterministic an oversized sample recurs at the
# same step -- which is how the DROID runs died at exactly step 100, three
# times. The merged set has three instructions, the longest being "Transfer the
# egg from the pan into the bowl." -- about four tokens above the shortest, well
# inside the headroom validate_dataloader.py measured. Raising the cap allocates
# nothing: text is padded to the longest item per batch, and
# max_sequence_length is purely a processor-side guard.
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-896}"

POLICY_PATH="${POLICY_PATH:-}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
IMAGE_TRANSFORMS_ENABLE="${IMAGE_TRANSFORMS_ENABLE:-true}"
# lerobot's default (1e-4s) is tight enough that float32 timestamp rounding can
# push a query away from its nearest decoded frame and raise a fatal
# FrameTimestampError. A 30 fps frame period is 33ms, so 1ms is comfortably
# inside it. Stage1 skips video decode and never hits this; Stage2 decodes.
# This default is 1e-3 to match both the comment above and train_stage2.sh,
# which exports the same value and would otherwise silently override a tighter
# default only on the stage that actually decodes video.
TOLERANCE_S="${TOLERANCE_S:-1e-3}"

JOB_NAME="${JOB_NAME:-molmoact2-yam}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/${JOB_NAME}/${RUN_ID}}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# Micro-batches per optimizer step. Global batch is
# BATCH_SIZE * n_gpus * GRAD_ACCUM_STEPS. STEPS counts optimizer steps, so the
# loop consumes STEPS * GRAD_ACCUM_STEPS batches. Requires the
# gradient-accumulation support in lerobot (feat/gradient-accumulation).
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
LOG_FREQ="${LOG_FREQ:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
SEED="${SEED:-1000}"

ACTION_MODE="${ACTION_MODE:-continuous}"
TRAIN_ACTION_EXPERT_ONLY="${TRAIN_ACTION_EXPERT_ONLY:-false}"
DISABLE_VISUAL_INPUT="${DISABLE_VISUAL_INPUT:-false}"
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
OPTIMIZER_VIT_LR="${OPTIMIZER_VIT_LR:-1e-5}"
OPTIMIZER_CONNECTOR_LR="${OPTIMIZER_CONNECTOR_LR:-1e-5}"
OPTIMIZER_ACTION_EXPERT_LR="${OPTIMIZER_ACTION_EXPERT_LR:-5e-5}"
OPTIMIZER_BETAS="${OPTIMIZER_BETAS:-[0.9,0.95]}"
OPTIMIZER_EPS="${OPTIMIZER_EPS:-1e-6}"
OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-0.0}"
OPTIMIZER_GRAD_CLIP_NORM="${OPTIMIZER_GRAD_CLIP_NORM:-1.0}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"
SCHEDULER_VLM_WARMUP_STEPS="${SCHEDULER_VLM_WARMUP_STEPS:-${SCHEDULER_WARMUP_STEPS}}"
SCHEDULER_VIT_WARMUP_STEPS="${SCHEDULER_VIT_WARMUP_STEPS:-${SCHEDULER_WARMUP_STEPS}}"
SCHEDULER_CONNECTOR_WARMUP_STEPS="${SCHEDULER_CONNECTOR_WARMUP_STEPS:-${SCHEDULER_WARMUP_STEPS}}"
SCHEDULER_ACTION_EXPERT_WARMUP_STEPS="${SCHEDULER_ACTION_EXPERT_WARMUP_STEPS:-500}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-${STEPS}}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-1e-5}"

WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-molmoact2-yam}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"

mkdir -p "$(dirname "${OUTPUT_DIR}")"
if [[ "${NUM_MACHINES}" -gt 1 ]]; then
  LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.console.rank${MACHINE_RANK}.log}"
else
  LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.console.log}"
fi
mkdir -p "$(dirname "${LOG_FILE}")"

if [[ -z "${RESUME_CHECKPOINT}" && "${RESUME_MODE}" == "auto" ]]; then
  candidate="${OUTPUT_DIR}/checkpoints/last"
  if [[ -f "${candidate}/pretrained_model/train_config.json" ]]; then
    RESUME_CHECKPOINT="${candidate}"
  fi
fi

RESUME_CONFIG_PATH=""
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  if [[ -f "${RESUME_CHECKPOINT}" ]]; then
    RESUME_CONFIG_PATH="${RESUME_CHECKPOINT}"
  elif [[ -f "${RESUME_CHECKPOINT}/train_config.json" ]]; then
    RESUME_CONFIG_PATH="${RESUME_CHECKPOINT}/train_config.json"
  elif [[ -f "${RESUME_CHECKPOINT}/pretrained_model/train_config.json" ]]; then
    RESUME_CONFIG_PATH="${RESUME_CHECKPOINT}/pretrained_model/train_config.json"
  else
    echo "Resume checkpoint has no train_config.json: ${RESUME_CHECKPOINT}" >&2
    exit 2
  fi
  RESUME_CHECKPOINT="$(cd "$(dirname "${RESUME_CONFIG_PATH}")/.." && pwd)"
  if [[ ! -d "${RESUME_CHECKPOINT}/training_state" ]]; then
    echo "Resume checkpoint has no training_state: ${RESUME_CHECKPOINT}" >&2
    exit 2
  fi
fi

echo "[yam] GPUs=${CUDA_VISIBLE_DEVICES} (local=${NUM_PROCESSES_LOCAL} total=${NUM_PROCESSES})"
echo "[yam] machines=${NUM_MACHINES} rank=${MACHINE_RANK} main=${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT}"
echo "[yam] dataset=${DATASET_REPO_ID} root=${DATASET_ROOT}"
echo "[yam] manifest=${SAMPLE_MANIFEST_PATH}"
echo "[yam] policy_path=${POLICY_PATH:-<fresh>} checkpoint=${CHECKPOINT_PATH}@${CHECKPOINT_REVISION}"
echo "[yam] output_dir=${OUTPUT_DIR}"
echo "[yam] chunk_size=${CHUNK_SIZE} (${CHUNK_SIZE} frames = $((CHUNK_SIZE / EXPECTED_FPS)).0s at ${EXPECTED_FPS} fps)"
echo "[yam] batch_size/gpu=${BATCH_SIZE} accum=${GRAD_ACCUM_STEPS} steps=${STEPS} (optimizer steps) seed=${SEED}"
echo "[yam] global batch=$((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))"
echo "[yam] action_mode=${ACTION_MODE} action_expert_only=${TRAIN_ACTION_EXPERT_ONLY} visual_disabled=${DISABLE_VISUAL_INPUT}"
echo "[yam] tolerance_s=${TOLERANCE_S} max_sequence_length=${MAX_SEQUENCE_LENGTH}"
echo "[yam] AdamW betas=${OPTIMIZER_BETAS} eps=${OPTIMIZER_EPS} weight_decay=${OPTIMIZER_WEIGHT_DECAY} grad_clip=${OPTIMIZER_GRAD_CLIP_NORM}"
echo "[yam] resume=${RESUME_CONFIG_PATH:-false}"

cd "${WS}/lerobot"
CMD=(
  accelerate launch
  --num_processes="${NUM_PROCESSES}"
  --num_machines="${NUM_MACHINES}"
  --machine_rank="${MACHINE_RANK}"
  --main_process_ip="${MAIN_PROCESS_IP}"
  --main_process_port="${MAIN_PROCESS_PORT}"
  --mixed_precision=bf16
  -m lerobot.scripts.lerobot_train
)

if [[ -n "${RESUME_CONFIG_PATH}" ]]; then
  CMD+=(
    --config_path="${RESUME_CONFIG_PATH}"
    --resume=true
    --output_dir="${OUTPUT_DIR}"
    --steps="${STEPS}"
    --gradient_accumulation_steps="${GRAD_ACCUM_STEPS}"
    --log_freq="${LOG_FREQ}"
    --save_freq="${SAVE_FREQ}"
    --eval_freq=-1
  )
else
  CMD+=(
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --dataset.sample_indices_path="${SAMPLE_MANIFEST_PATH}"
    --dataset.video_backend="${VIDEO_BACKEND}"
    --dataset.image_transforms.enable="${IMAGE_TRANSFORMS_ENABLE}"
    --tolerance_s="${TOLERANCE_S}"
    --policy.device=cuda
    --policy.action_mode="${ACTION_MODE}"
    --policy.max_sequence_length="${MAX_SEQUENCE_LENGTH}"
    --policy.chunk_size="${CHUNK_SIZE}"
    --policy.n_action_steps="${CHUNK_SIZE}"
    --policy.setup_type="bimanual yam robot arms"
    --policy.control_mode="absolute joint pose"
    --policy.image_keys='["observation.images.top","observation.images.left","observation.images.right"]'
    --policy.model_dtype=bfloat16
    --policy.num_flow_timesteps=8
    --policy.gradient_checkpointing=true
    --policy.freeze_embedding=true
    --policy.train_action_expert_only="${TRAIN_ACTION_EXPERT_ONLY}"
    --policy.disable_visual_input="${DISABLE_VISUAL_INPUT}"
    --policy.normalize_gripper=false
    --policy.enable_knowledge_insulation=false
    --policy.optimizer_lr="${OPTIMIZER_LR}"
    --policy.optimizer_vit_lr="${OPTIMIZER_VIT_LR}"
    --policy.optimizer_connector_lr="${OPTIMIZER_CONNECTOR_LR}"
    --policy.optimizer_action_expert_lr="${OPTIMIZER_ACTION_EXPERT_LR}"
    --policy.optimizer_betas="${OPTIMIZER_BETAS}"
    --policy.optimizer_eps="${OPTIMIZER_EPS}"
    --policy.optimizer_weight_decay="${OPTIMIZER_WEIGHT_DECAY}"
    --policy.optimizer_grad_clip_norm="${OPTIMIZER_GRAD_CLIP_NORM}"
    --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}"
    --policy.scheduler_vlm_warmup_steps="${SCHEDULER_VLM_WARMUP_STEPS}"
    --policy.scheduler_vit_warmup_steps="${SCHEDULER_VIT_WARMUP_STEPS}"
    --policy.scheduler_connector_warmup_steps="${SCHEDULER_CONNECTOR_WARMUP_STEPS}"
    --policy.scheduler_action_expert_warmup_steps="${SCHEDULER_ACTION_EXPERT_WARMUP_STEPS}"
    --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}"
    --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}"
    --policy.push_to_hub=false
    --job_name="${JOB_NAME}"
    --output_dir="${OUTPUT_DIR}"
    --steps="${STEPS}"
    --seed="${SEED}"
    --resume=false
    --batch_size="${BATCH_SIZE}"
    --gradient_accumulation_steps="${GRAD_ACCUM_STEPS}"
    --num_workers="${NUM_WORKERS}"
    --log_freq="${LOG_FREQ}"
    --eval_freq=-1
    --save_checkpoint="${SAVE_CHECKPOINT}"
    --save_freq="${SAVE_FREQ}"
  )

  if [[ -n "${POLICY_PATH}" ]]; then
    CMD+=(--policy.path="${POLICY_PATH}")
  else
    CMD+=(
      --policy.type=molmoact2
      --policy.checkpoint_path="${CHECKPOINT_PATH}"
      --policy.checkpoint_revision="${CHECKPOINT_REVISION}"
    )
  fi

  if [[ "${WANDB_ENABLE}" == "true" ]]; then
    CMD+=(--wandb.enable=true --wandb.project="${WANDB_PROJECT}")
    if [[ -n "${WANDB_ENTITY}" ]]; then
      CMD+=(--wandb.entity="${WANDB_ENTITY}")
    fi
    if [[ "${WANDB_DISABLE_ARTIFACT}" == "true" ]]; then
      CMD+=(--wandb.disable_artifact=true)
    fi
  else
    CMD+=(--wandb.enable=false)
  fi
fi

CMD+=("${PASSTHROUGH_ARGS[@]}")
printf '[yam] running:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  exit 0
fi

set -o pipefail
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
