#!/usr/bin/env bash
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${WS}/scripts/activate_train_env.sh"

DEFAULT_POLICY_PATH="${WS}/lerobot/outputs/libero_goal_prior_v2b/seed_1000/stage2/checkpoints/005000/pretrained_model"
POLICY_PATH="${1:-${POLICY_PATH:-${DEFAULT_POLICY_PATH}}}"
if [[ -z "${POLICY_PATH}" || ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "Goal-pose v2b checkpoint not found: ${POLICY_PATH}" >&2
  echo "Override with: bash $0 /path/to/checkpoint/pretrained_model" >&2
  exit 2
fi
POLICY_PATH="$(cd "${POLICY_PATH}" && pwd)"

python - "${POLICY_PATH}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = json.load(f)
expected = {
    "type": "molmoact2",
    "enable_goal_pose": True,
    "goal_conditioning_mode": "semantic_visual_recurrent",
    "goal_token_source": "learnable_queries",
    "num_semantic_visual_tokens": 100,
    "num_semantic_visual_pose_tokens": 8,
    "semantic_visual_hidden_dim": 768,
    "semantic_visual_enable_self_attention": True,
    "semantic_visual_num_layer_groups": 6,
    "mask_image_from_action_expert": True,
    "enable_pose_reconstruction": True,
}
mismatches = [
    f"{key}: expected {want!r}, got {cfg.get(key)!r}"
    for key, want in expected.items()
    if cfg.get(key) != want
]
pose_tokens = cfg.get("num_semantic_visual_pose_tokens")
if not (isinstance(pose_tokens, int) and 1 <= pose_tokens < cfg.get("num_semantic_visual_tokens", 0)):
    mismatches.append(
        "num_semantic_visual_pose_tokens: expected 1 <= P < num_semantic_visual_tokens, "
        f"got {pose_tokens!r}"
    )
if mismatches:
    raise SystemExit(
        "Refusing to evaluate: checkpoint is not Goal-Pose Prior v2b (scheme-2a):\n  "
        + "\n  ".join(mismatches)
    )
print(
    "[eval] verified v2b:"
    f" mode={cfg['goal_conditioning_mode']}"
    f" tokens={cfg['num_semantic_visual_tokens']}"
    f" pose_tokens={pose_tokens}"
    f" groups={cfg['semantic_visual_num_layer_groups']}"
    f" self_attn={cfg['semantic_visual_enable_self_attention']}"
    f" hidden={cfg['semantic_visual_hidden_dim']}"
)
PY

EVAL_SEED="${EVAL_SEED:-1000}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-${EPISODES_PER_TASK}}"
GPU_IDS=(${EVAL_GPU_IDS:-7})
SUITES=(libero_spatial libero_object libero_10 libero_goal)
RESOURCE_ROOT="${LIBERO_RESOURCE_ROOT:-/data2/JM/Code/molmo_serious/molmoact2-main}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${RESOURCE_ROOT}/.cache/libero_config}"
export HF_HOME="${HF_HOME:-${RESOURCE_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${RESOURCE_ROOT}/.cache/hf_datasets}"
export TORCH_HOME="${TORCH_HOME:-${RESOURCE_ROOT}/.cache/torch}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="${WS}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

if not hasattr(MolmoAct2Policy, "_uses_policy_continuous_generation"):
    raise SystemExit(
        "Refusing to evaluate: local MolmoAct2 code lacks the non-RTC learned-context "
        "inference fix."
    )
print("[eval] verified non-RTC learned-context inference fix")
PY

checkpoint_step="$(basename "$(dirname "${POLICY_PATH}")")"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-goal_prior_v2b_${checkpoint_step}}"
EVAL_ROOT="${EVAL_ROOT:-${WS}/lerobot/outputs/libero_eval/${CHECKPOINT_LABEL}/libero_seed_${EVAL_SEED}}"
mkdir -p "${EVAL_ROOT}"

cat >"${EVAL_ROOT}/run_manifest.json" <<EOF
{
  "benchmark": "libero",
  "policy_path": "${POLICY_PATH}",
  "checkpoint_label": "${CHECKPOINT_LABEL}",
  "model": "goal_pose_prior_v2b",
  "suites": ["libero_spatial", "libero_object", "libero_10", "libero_goal"],
  "episodes_per_task": ${EPISODES_PER_TASK},
  "max_episodes_rendered": ${MAX_EPISODES_RENDERED},
  "eval_seed": ${EVAL_SEED},
  "gpu_ids": "${GPU_IDS[*]}"
}
EOF

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[dry-run] policy=${POLICY_PATH}"
  echo "[dry-run] output=${EVAL_ROOT}"
  echo "[dry-run] suites=${SUITES[*]} episodes/task=${EPISODES_PER_TASK} gpus=${GPU_IDS[*]}"
  exit 0
fi

pids=()
run_sequential=false
if [[ "${#GPU_IDS[@]}" -eq 1 ]]; then
  run_sequential=true
  echo "[eval] single GPU ${GPU_IDS[0]}: suites will run sequentially"
fi

for index in "${!SUITES[@]}"; do
  suite="${SUITES[$index]}"
  gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
  output_dir="${EVAL_ROOT}/${suite}"
  mkdir -p "${output_dir}"
  if [[ -f "${output_dir}/eval_info.json" ]]; then
    echo "Refusing to overwrite completed evaluation: ${output_dir}" >&2
    exit 2
  fi
  (
    printf '{"status":"running","gpu":"%s"}\n' "${gpu}" >"${output_dir}/process_status.json"
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" \
      python -m lerobot.scripts.lerobot_eval \
        --policy.path="${POLICY_PATH}" \
        --policy.device=cuda \
        --policy.inference_action_mode=continuous \
        --policy.disable_visual_input=false \
        --policy.per_episode_seed=true \
        --policy.eval_seed="${EVAL_SEED}" \
        --policy.enable_inference_cuda_graph=false \
        --env.type=libero \
        --env.task="${suite}" \
        --env.control_mode=relative \
        --env.max_parallel_tasks=1 \
        --eval.n_episodes="${EPISODES_PER_TASK}" \
        --eval.batch_size="${EVAL_BATCH_SIZE}" \
        --eval.max_episodes_rendered="${MAX_EPISODES_RENDERED}" \
        --eval.use_async_envs=true \
        --seed="${EVAL_SEED}" \
        --output_dir="${output_dir}" \
        >"${output_dir}/eval.log" 2>&1
    rc=$?
    if [[ "${rc}" -eq 0 ]]; then status=final; else status=failed; fi
    printf '{"status":"%s","exit_code":%d,"gpu":"%s"}\n' \
      "${status}" "${rc}" "${gpu}" >"${output_dir}/process_status.json"
    exit "${rc}"
  ) &
  pids+=("$!")
  if [[ "${run_sequential}" == "true" ]]; then
    if ! wait "${pids[-1]}"; then
      echo "[eval] suite ${suite} failed" >&2
    fi
  fi
done

if [[ "${run_sequential}" != "true" ]]; then
  python "${WS}/scripts/libero_eval/monitor_eval.py" --eval-root "${EVAL_ROOT}"
fi

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then rc=1; fi
done
python "${WS}/scripts/libero_eval/monitor_eval.py" --eval-root "${EVAL_ROOT}" --once
echo "[eval] results: ${EVAL_ROOT}"
exit "${rc}"
