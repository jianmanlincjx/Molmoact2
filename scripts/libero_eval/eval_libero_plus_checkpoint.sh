#!/usr/bin/env bash
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${WS}/scripts/activate_train_env.sh"

POLICY_PATH="${1:-${POLICY_PATH:-}}"
if [[ -z "${POLICY_PATH}" || ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "Usage: bash $0 /path/to/checkpoint/pretrained_model" >&2
  exit 2
fi
POLICY_PATH="$(cd "${POLICY_PATH}" && pwd)"

EVAL_SEED="${EVAL_SEED:-1000}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-${EPISODES_PER_TASK}}"
SAMPLES_PER_CELL="${SAMPLES_PER_CELL:-2}"
PLUS_PROTOCOL="${PLUS_PROTOCOL:-base_category}"
GPU_IDS=(${EVAL_GPU_IDS:-0 1 2 3})
SUITES=(libero_object libero_10 libero_goal libero_spatial)
RESOURCE_ROOT="${LIBERO_RESOURCE_ROOT:-/data2/JM/Code/molmo_serious/molmoact2-main}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-${RESOURCE_ROOT}/third_party/LIBERO-plus}"
LIBERO_PLUS_PACKAGE="${LIBERO_PLUS_ROOT}/libero/libero"
CLASSIFICATION="${LIBERO_PLUS_CLASSIFICATION:-${LIBERO_PLUS_PACKAGE}/benchmark/task_classification.json}"

for required in \
  "${LIBERO_PLUS_ROOT}/libero/__init__.py" \
  "${LIBERO_PLUS_PACKAGE}/assets" \
  "${LIBERO_PLUS_PACKAGE}/init_files" \
  "${CLASSIFICATION}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing LIBERO-Plus resource: ${required}" >&2
    exit 2
  fi
done

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${RESOURCE_ROOT}/.cache/libero_config_plus}"
export HF_HOME="${HF_HOME:-${RESOURCE_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${RESOURCE_ROOT}/.cache/hf_datasets}"
export TORCH_HOME="${TORCH_HOME:-${RESOURCE_ROOT}/.cache/torch}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# The Plus fork must precede site-packages while current LeRobot remains the source of policy/eval code.
export PYTHONPATH="${LIBERO_PLUS_ROOT}:${WS}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"

checkpoint_step="$(basename "$(dirname "${POLICY_PATH}")")"
run_name="$(basename "$(dirname "$(dirname "$(dirname "${POLICY_PATH}")")")")"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-${run_name}_${checkpoint_step}}"
EVAL_ROOT="${EVAL_ROOT:-${WS}/lerobot/outputs/libero_eval/${CHECKPOINT_LABEL}/libero_plus_balanced_seed_${EVAL_SEED}}"
mkdir -p "${EVAL_ROOT}"

python "${WS}/scripts/libero_eval/build_libero_plus_manifest.py" \
  --classification "${CLASSIFICATION}" \
  --output "${EVAL_ROOT}/task_manifest.json" \
  --seed "${EVAL_SEED}" \
  --protocol "${PLUS_PROTOCOL}" \
  --samples-per-cell "${SAMPLES_PER_CELL}" \
  --episodes-per-task "${EPISODES_PER_TASK}"

cat >"${EVAL_ROOT}/run_manifest.json" <<EOF
{
  "benchmark": "libero_plus",
  "protocol": "${PLUS_PROTOCOL}",
  "samples_per_cell": ${SAMPLES_PER_CELL},
  "policy_path": "${POLICY_PATH}",
  "checkpoint_label": "${CHECKPOINT_LABEL}",
  "episodes_per_task": ${EPISODES_PER_TASK},
  "eval_seed": ${EVAL_SEED},
  "gpu_ids": "${GPU_IDS[*]}",
  "excluded_categories": ["Language Instructions"]
}
EOF

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[dry-run] policy=${POLICY_PATH}"
  echo "[dry-run] output=${EVAL_ROOT}"
  echo "[dry-run] protocol=${PLUS_PROTOCOL} samples/cell=${SAMPLES_PER_CELL} episodes/task=${EPISODES_PER_TASK}"
  echo "[dry-run] suites=${SUITES[*]} gpus=${GPU_IDS[*]}"
  exit 0
fi

pids=()
for index in "${!SUITES[@]}"; do
  suite="${SUITES[$index]}"
  gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
  output_dir="${EVAL_ROOT}/${suite}"
  mkdir -p "${output_dir}"
  if [[ -f "${output_dir}/eval_info.json" ]]; then
    echo "Refusing to overwrite completed evaluation: ${output_dir}" >&2
    exit 2
  fi
  task_ids="$(
    python - "${EVAL_ROOT}/task_manifest.json" "${suite}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
ids = [task["task_id"] for task in manifest["tasks"] if task["suite"] == sys.argv[2]]
print("[" + ",".join(str(task_id) for task_id in ids) + "]")
PY
  )"
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
        --env.type=libero_plus \
        --env.task="${suite}" \
        --env.task_ids="${task_ids}" \
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
done

python "${WS}/scripts/libero_eval/monitor_eval.py" --eval-root "${EVAL_ROOT}"

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then rc=1; fi
done
python "${WS}/scripts/libero_eval/monitor_eval.py" --eval-root "${EVAL_ROOT}" --once
echo "[eval] results: ${EVAL_ROOT}"
exit "${rc}"
