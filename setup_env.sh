#!/usr/bin/env bash
# Create a local virtual environment for kernelMO with CUDA-enabled PyTorch.
#
# Usage:
#   bash setup_env.sh            # core env + CUDA PyTorch
#   bash setup_env.sh --cpu      # core env + CPU-only PyTorch
#   bash setup_env.sh --optional # also install optuna / skopt / neuraloperator
#
# The H100 nodes here run driver 560.35 (CUDA 12.x), so the default uses the
# cu124 PyTorch wheels. Override the index with TORCH_INDEX_URL if needed.
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
INSTALL_OPTIONAL=0
INSTALL_CPU=0

for arg in "$@"; do
  case "$arg" in
    --optional) INSTALL_OPTIONAL=1 ;;
    --cpu) INSTALL_CPU=1 ; TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu" ;;
    *) echo "Unknown option: $arg" ; exit 1 ;;
  esac
done

echo ">> Creating virtual environment in ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

echo ">> Installing PyTorch from ${TORCH_INDEX_URL}"
python -m pip install torch --index-url "${TORCH_INDEX_URL}"

echo ">> Installing core requirements"
python -m pip install -r requirements.txt

if [[ "${INSTALL_OPTIONAL}" -eq 1 ]]; then
  echo ">> Installing optional requirements"
  python -m pip install -r requirements-optional.txt
fi

echo ">> Registering Jupyter kernel 'kernelmo'"
python -m ipykernel install --user --name kernelmo --display-name "Python (kernelMO)" || true

echo ">> Verifying install"
python - <<'PY'
import numpy, sklearn, h5py, pandas, matplotlib
print("numpy", numpy.__version__, "| sklearn", sklearn.__version__,
      "| h5py", h5py.__version__, "| pandas", pandas.__version__)
try:
    import torch
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
          "| device count:", torch.cuda.device_count())
except Exception as exc:  # noqa: BLE001
    print("torch import failed:", exc)
PY

echo
echo "Done. Activate with:  source ${VENV_DIR}/bin/activate"
