#!/usr/bin/env bash
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${WS}/scripts/activate_train_env.sh"

SEED="${SEED:-1000}"
DATASET_ROOT="${DATASET_ROOT:-/data2/JM/dataset/libero_lerobot_format}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/libero_lerobot_format}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${WS}/lerobot/outputs/libero_two_stage/seed_${SEED}}"
OUTPUT_DIR="${CANONICAL_INIT_DIR:-${EXPERIMENT_ROOT}/00_canonical_init}"
MANIFEST_PATH="${OUTPUT_DIR}.manifest.json"

CMD=(
  python "${WS}/scripts/libero_two_stage/create_canonical_init.py"
  --dataset-root "${DATASET_ROOT}"
  --dataset-repo-id "${DATASET_REPO_ID}"
  --molmoact2-checkpoint "${WS}/Checkpoint/MolmoAct2"
  --vlm-checkpoint "${WS}/Checkpoint/Molmo2-ER"
  --output-dir "${OUTPUT_DIR}"
  --seed "${SEED}"
)

python "${WS}/scripts/libero_two_stage/write_manifest.py" \
  --workspace "${WS}" \
  --dataset-root "${DATASET_ROOT}" \
  --output "${MANIFEST_PATH}" \
  --run-type canonical_init \
  --seed "${SEED}" \
  --command "${CMD[@]}"

"${CMD[@]}"
cp "${MANIFEST_PATH}" "${OUTPUT_DIR}/experiment_manifest.json"
