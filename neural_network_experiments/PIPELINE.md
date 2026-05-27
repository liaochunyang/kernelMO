# Comparison Pipeline: How to Run and Check Results

## How to run

### 0. One-time: activate env (already built)

```bash
cd /home/jingmins/kernelMO
source .venv/bin/activate
```

### 1. Pick a free GPU (nodes are shared — check first)

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
```

Prefix commands with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<free idx>`.

### 2. Quick sanity run (~1 min: one PDE, one model)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python neural_network_experiments/run_comparison.py \
  --frameworks Framework2 --pdes Conservation_law --models mionet \
  --epochs 1 --output /tmp/sanity.csv
```

### 3. Full comparison (4 settings x 2 frameworks x 5 PDEs, all OOD files)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python neural_network_experiments/run_comparison.py \
  --epochs 50 --batch-size 64 \
  --output neural_network_experiments/results_comparison.csv
```

### 4. Add the network-size sweep (small/medium/large — ~3x runtime)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python neural_network_experiments/run_comparison.py \
  --sizes small medium large --epochs 50 \
  --output neural_network_experiments/results_comparison.csv
```

The driver writes the CSV incrementally after each run, so you can monitor progress (or interrupt)
safely. Data root defaults to `/home/shared/dataset/KernelMOL`.

## How to check the results

### A. Quick peek at the CSV

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('neural_network_experiments/results_comparison.csv')
print(df[df.split=='test'].pivot_table(index=['framework','pde'],
      columns='setting', values='mean_relative_error', aggfunc='min').to_string())
PY
```

### B. Full report (kernel vs. neural, one table) — open the notebook with the `Python (kernelMO)` kernel:

```bash
jupyter lab neural_network_experiments/comparison_visualize.ipynb
```

It renders: in-distribution test table, OOD table by `ood_file`, per-framework bar charts,
training/inference-time tables, and the accuracy-vs-size sweep.
