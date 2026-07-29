#!/usr/bin/env bash
# Evaluate the 30k clean-bootstrap baseline on four LIBERO suites in parallel.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${WS}/scripts/activate_train_env.sh"

DEFAULT_POLICY_PATH="${WS}/lerobot/outputs/libero_goal_prior/seed_1000/libero_baseline/checkpoints/030000/pretrained_model"
POLICY_PATH="${1:-${POLICY_PATH:-${DEFAULT_POLICY_PATH}}}"
if [[ -z "${POLICY_PATH}" || ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "LIBERO baseline checkpoint not found: ${POLICY_PATH}" >&2
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
    "action_mode": "continuous",
    "enable_goal_pose": False,
    "disable_visual_input": False,
    "mask_image_from_action_expert": False,
    "enable_pose_reconstruction": False,
    "chunk_size": 10,
    "n_action_steps": 10,
}
mismatches = [
    f"{key}: expected {want!r}, got {cfg.get(key)!r}"
    for key, want in expected.items()
    if cfg.get(key) != want
]
if mismatches:
    raise SystemExit(
        "Refusing to evaluate: checkpoint is not the clean LIBERO baseline:\n  "
        + "\n  ".join(mismatches)
    )
print(
    "[eval] verified baseline:"
    f" goal={cfg['enable_goal_pose']}"
    f" visual_disabled={cfg['disable_visual_input']}"
    f" chunk={cfg['chunk_size']}"
    f" pretrained_path={cfg.get('pretrained_path')!r}"
)
PY

EVAL_SEED="${EVAL_SEED:-1000}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-${EPISODES_PER_TASK}}"
GPU_IDS=(${EVAL_GPU_IDS:-0 1 2 3})
SUITES=(libero_spatial libero_goal libero_object libero_10)

if [[ "${#GPU_IDS[@]}" -ne "${#SUITES[@]}" ]]; then
  echo "Baseline evaluation requires exactly four GPU IDs (spatial/goal/object/10)." >&2
  echo "Got: ${GPU_IDS[*]}" >&2
  exit 2
fi

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

checkpoint_step="$(basename "$(dirname "${POLICY_PATH}")")"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-libero_baseline_${checkpoint_step}}"
EVAL_ROOT="${EVAL_ROOT:-${WS}/lerobot/outputs/libero_eval/${CHECKPOINT_LABEL}/libero_seed_${EVAL_SEED}}"
mkdir -p "${EVAL_ROOT}"

cat >"${EVAL_ROOT}/run_manifest.json" <<EOF
{
  "benchmark": "libero",
  "policy_path": "${POLICY_PATH}",
  "checkpoint_label": "${CHECKPOINT_LABEL}",
  "model": "clean_bootstrap_baseline",
  "suites": ["libero_spatial", "libero_goal", "libero_object", "libero_10"],
  "suite_gpu_mapping": {
    "libero_spatial": "${GPU_IDS[0]}",
    "libero_goal": "${GPU_IDS[1]}",
    "libero_object": "${GPU_IDS[2]}",
    "libero_10": "${GPU_IDS[3]}"
  },
  "episodes_per_task": ${EPISODES_PER_TASK},
  "max_episodes_rendered": ${MAX_EPISODES_RENDERED},
  "eval_seed": ${EVAL_SEED}
}
EOF

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[dry-run] policy=${POLICY_PATH}"
  echo "[dry-run] output=${EVAL_ROOT}"
  for index in "${!SUITES[@]}"; do
    echo "[dry-run] ${SUITES[$index]} -> GPU ${GPU_IDS[$index]}"
  done
  echo "[dry-run] episodes/task=${EPISODES_PER_TASK} batch=${EVAL_BATCH_SIZE}"
  exit 0
fi

pids=()
for index in "${!SUITES[@]}"; do
  suite="${SUITES[$index]}"
  gpu="${GPU_IDS[$index]}"
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
  echo "[eval] launched ${suite} on GPU ${gpu} (pid=${pids[-1]})"
done

python "${WS}/scripts/libero_eval/monitor_eval.py" --eval-root "${EVAL_ROOT}"

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then rc=1; fi
done
python "${WS}/scripts/libero_eval/monitor_eval.py" --eval-root "${EVAL_ROOT}" --once
echo "[eval] results: ${EVAL_ROOT}"
exit "${rc}"
