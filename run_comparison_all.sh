#!/usr/bin/env bash
# End-to-end operator-learning comparison: env -> free GPU -> train -> summarize.
#
# Usage:
#   bash run_comparison_all.sh                 # full comparison (medium size, 50 epochs)
#   bash run_comparison_all.sh --quick         # 1-PDE / 1-model sanity run (~1 min)
#   bash run_comparison_all.sh --sweep         # size sweep small/medium/large (~3x runtime)
#   bash run_comparison_all.sh --setup         # (re)build the venv first, then run
#
# Env overrides:
#   EPOCHS=50 BATCH=64 GPU=0 SIZES="small medium large" OUTPUT=path/to.csv \
#     bash run_comparison_all.sh
#
# Extra args after `--` are forwarded to run_comparison.py, e.g.:
#   bash run_comparison_all.sh -- --frameworks Framework2 --pdes Conservation_law
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

VENV_DIR="${VENV_DIR:-.venv}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-64}"
SIZES="${SIZES:-medium}"
OUTPUT="${OUTPUT:-}"   # resolved per-mode below so --quick doesn't clobber the real CSV
CKPT_DIR="${CKPT_DIR:-neural_network_experiments/checkpoints}"
DO_SETUP=0
QUICK=0
EXTRA_ARGS=()

# ---- parse flags -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup) DO_SETUP=1 ; shift ;;
    --quick) QUICK=1 ; shift ;;
    --sweep) SIZES="small medium large" ; shift ;;
    --) shift ; EXTRA_ARGS=("$@") ; break ;;
    *) echo "Unknown option: $1" ; exit 1 ;;
  esac
done

# ---- 0. environment --------------------------------------------------------
if [[ "${DO_SETUP}" -eq 1 || ! -d "${VENV_DIR}" ]]; then
  echo ">> Building virtual environment"
  bash setup_env.sh
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ---- 1. pick a free GPU ----------------------------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  # choose the GPU index with the most free memory
  GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d ' ')"
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi
echo ">> Using GPU index: ${CUDA_VISIBLE_DEVICES:-cpu}"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || true

# ---- 2. run the comparison -------------------------------------------------
if [[ "${QUICK}" -eq 1 ]]; then
  OUTPUT="${OUTPUT:-/tmp/sanity.csv}"
  echo ">> Quick sanity run (Framework2 / Conservation_law / MIONet, 1 epoch) -> ${OUTPUT}"
  python neural_network_experiments/run_comparison.py \
    --frameworks Framework2 --pdes Conservation_law --models mionet \
    --epochs 1 --output "${OUTPUT}" --checkpoint-dir "${CKPT_DIR}" "${EXTRA_ARGS[@]}"
else
  OUTPUT="${OUTPUT:-neural_network_experiments/results_comparison.csv}"
  echo ">> Full comparison (sizes: ${SIZES}, epochs: ${EPOCHS}, checkpoints: ${CKPT_DIR})"
  python neural_network_experiments/run_comparison.py \
    --sizes ${SIZES} --epochs "${EPOCHS}" --batch-size "${BATCH}" \
    --output "${OUTPUT}" --checkpoint-dir "${CKPT_DIR}" "${EXTRA_ARGS[@]}"
fi

# ---- 3. summarize ----------------------------------------------------------
echo
echo ">> Test mean relative error (lower is better):"
python - "${OUTPUT}" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
test = df[df["split"] == "test"]
if test.empty:
    print("(no test rows)")
else:
    piv = test.pivot_table(index=["framework", "pde"], columns="setting",
                           values="mean_relative_error", aggfunc="min")
    print(piv.to_string(float_format=lambda v: f"{v:.3e}"))
PY

echo
echo "Done. Results CSV: ${OUTPUT}"
echo "Open the full report:  jupyter lab neural_network_experiments/comparison_visualize.ipynb"
