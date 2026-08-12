# Source before any uv / training commands in this repo.
# Caches default to /scratch (large quota) rather than the NFS home
# directory (40GB quota, easily blown by the HF `datasets` Arrow cache built
# from the full ~17M-anchor derived dataset). Override CACHE_ROOT to move
# them elsewhere.

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${CACHE_ROOT:-/scratch/shailesh.xml/.cache/Molmoact2}"

export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-16}"
# Do not inject a secondary pytorch wheel index first — it breaks cmake resolution.
unset UV_EXTRA_INDEX_URL UV_INDEX_URL

export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TMPDIR="${CACHE_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"

# Prefer HF mirror when Hub is slow (optional).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

echo "[env] WS=$WS"
echo "[env] UV_CACHE_DIR=$UV_CACHE_DIR"
echo "[env] HF_HOME=$HF_HOME"
echo "[env] TMPDIR=$TMPDIR"
