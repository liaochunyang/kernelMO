#!/usr/bin/env bash
# Poll for a free GPU, then run the resumable comparison sweep pinned to it.
#
# Every POLL_SECONDS it checks nvidia-smi. A GPU counts as "available" when it
# is both idle (utilization <= UTIL_MAX) and mostly empty (free memory >=
# FREE_MEM_MIN MiB). The first such GPU is picked; if none qualify, it waits and
# checks again. Once a GPU is found it launches:
#
#     EPOCHS=<EPOCHS> GPU=<idx> bash run_comparison_all.sh --sweep -- --skip-existing
#
# --skip-existing makes the run resume from results_comparison.csv, so only
# combos that have not been trained yet actually run.
#
# Usage:
#   bash run_when_gpu_free.sh
#
# Env overrides (with defaults):
#   POLL_SECONDS=60        # seconds between checks
#   UTIL_MAX=20            # max GPU utilization (%) to count as idle
#   FREE_MEM_MIN=40000     # min free memory (MiB) required
#   EPOCHS=200             # forwarded to run_comparison_all.sh
#   EXTRA="..."            # extra args appended after --skip-existing
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

POLL_SECONDS="${POLL_SECONDS:-60}"
UTIL_MAX="${UTIL_MAX:-20}"
FREE_MEM_MIN="${FREE_MEM_MIN:-40000}"
EPOCHS="${EPOCHS:-200}"
EXTRA="${EXTRA:-}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Echo the index of the first idle+empty GPU, or nothing if none qualify.
find_free_gpu() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.free \
             --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v u="${UTIL_MAX}" -v m="${FREE_MEM_MIN}" \
        '{ gsub(/ /,""); if ($2+0 <= u && $3+0 >= m) { print $1; exit } }'
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot detect GPUs." >&2
  exit 1
fi

echo ">> Waiting for a GPU with util <= ${UTIL_MAX}% and free mem >= ${FREE_MEM_MIN} MiB"
echo ">> Polling every ${POLL_SECONDS}s. Ctrl-C to stop."

while true; do
  GPU_IDX="$(find_free_gpu)"
  if [[ -n "${GPU_IDX}" ]]; then
    echo ">> [$(date '+%F %T')] GPU ${GPU_IDX} is free -> launching sweep (epochs=${EPOCHS})"
    GPU="${GPU_IDX}" EPOCHS="${EPOCHS}" \
      bash run_comparison_all.sh --sweep -- --skip-existing ${EXTRA}
    echo ">> [$(date '+%F %T')] Sweep finished on GPU ${GPU_IDX}."
    exit 0
  fi
  echo "   [$(date '+%F %T')] no free GPU; checking again in ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done
