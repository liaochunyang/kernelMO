# Neural Network Experiments

This folder contains PyTorch experiments for sample-mode PDE learning:

```text
[coefficients, initial condition] -> full solution trajectory
```

The data loader follows the same sample-style split used for Framework2/product-kernel experiments:

```python
u0 = dataset[:, INPUT_TIME_INDEX, :, CHANNEL_INDEX]
y  = dataset[:, :, :, CHANNEL_INDEX]
c  = coeffs
```

## Models

- `neuralop_fno`: an official `neuralop.models.FNO` wrapper over the `(t, x)` grid. The input channels are `u0(x)` broadcast across time, broadcast coefficient values, and `(t, x)` coordinates.
- `fno`: a lightweight local 1D FNO fallback over the spatial grid. Useful for debugging, but prefer `neuralop_fno` for serious FNO benchmarks.
- `deeponet`: a tensorized DeepONet matching the implementation in `main.ipynb`; the branch input is `[coefficients, u0]`, and the trunk input is `(t, x)`. (Comparison setting #2.)
- `deeponet_nocoef`: the same tensorized DeepONet but the branch sees only `u0` — the coefficients are ignored. (Comparison setting #1.)
- `mionet`: a MIONet with two branch networks (`u0` and `coefficients`) merged by an element-wise product and contracted with a shared `(t, x)` trunk. (Comparison setting #3.)
- `mno`: the MNO-style tensorized model from `main.ipynb`; it uses separate trunk `(t, x)`, branch `u0`, and leaf `coefficients` networks. (Comparison setting #4.)

## Full comparison driver

`run_comparison.py` trains all four comparison settings across both frameworks and every PDE,
writing one tidy CSV (`results_comparison.csv`). The split matches the kernel-method experiments
exactly (first-N, no shuffle): Framework2 uses the first 10000 samples for training and the next
4000 for testing; Framework1 uses the first 80% for training and the last 20% for testing. OOD rows
are added automatically wherever an `ood.h5` is present.

The `.h5` data is not in the repo. It lives at `/home/shared/dataset/KernelMOL`
(`Framework{1,2}/<pde>/dataset_simple/{solutions,ood}.h5`), which is the default `--data-root`.
Override with `--data-root` (or `--data-root .` if you have local copies under the repo).

```bash
python neural_network_experiments/run_comparison.py --epochs 50 --batch-size 64
# subset example:
python neural_network_experiments/run_comparison.py --frameworks Framework2 --pdes Conservation_law --epochs 5
# sweep three network sizes (small/medium/large):
python neural_network_experiments/run_comparison.py --sizes small medium large
# pin to a free GPU (nodes are shared):
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python neural_network_experiments/run_comparison.py
```

Each row records `mean_relative_error`, `train_time_seconds`, `predict_time_seconds`, and `n_params`,
and is tagged with `setting`, `size`, `split`, and `ood_file`. `--sizes` scales `num_trunk`/`num_branch`
(DeepONet/MNO), `latent_dim` (MIONet), and the MLP `hidden_dim` together; the default is `medium` only.

By default (`--ood-filenames auto`) the driver evaluates **every** `ood*.h5` present for a PDE, one
`split=ood` row per file (tagged by `ood_file`). This is how the parametric PDEs — which ship only
variant files like `ood_par_scale.h5`, no plain `ood.h5` — get OOD rows. Restrict with e.g.
`--ood-filenames ood.h5 ood_par_scale.h5`, or skip OOD with `--no-ood`.

Open `comparison_visualize.ipynb` for the combined report. It splices the kernel-method results
(`Framework{1,2}/results_*_no_pca.csv`, best of RBF/Matern per PDE) next to the four neural settings
in one table, then shows in/out-of-distribution error, bar charts, training/inference time, and an
accuracy-vs-size sweep. Run the notebook with the `Python (kernelMO)` Jupyter kernel.

## Quick Start

Install dependencies in the environment where `h5py` and PyTorch are available:

```bash
pip install torch h5py numpy pandas
pip install neuraloperator  # optional, recommended for official FNO
```

Run a small FNO test:

```bash
python neural_network_experiments/train_network.py \
  --model neuralop_fno \
  --framework Framework2 \
  --pde Conservation_law \
  --epochs 5 \
  --batch-size 16 \
  --train-size 1000 \
  --output neural_network_experiments/results_fno_test.csv
```

Run a small DeepONet test:

```bash
python neural_network_experiments/train_network.py \
  --model deeponet \
  --framework Framework2 \
  --pde Conservation_law \
  --epochs 5 \
  --batch-size 16 \
  --train-size 1000 \
  --trunk-batch-size 1024 \
  --output neural_network_experiments/results_deeponet_test.csv
```

Run a small MNO test with a similar branch-output size to the default DeepONet:

```bash
python neural_network_experiments/train_network.py \
  --model mno \
  --framework Framework2 \
  --pde Conservation_law \
  --epochs 5 \
  --batch-size 16 \
  --train-size 1000 \
  --num-trunk 100 \
  --num-branch 50 \
  --num-leaf 2 \
  --trunk-batch-size 1024 \
  --output neural_network_experiments/results_mno_test.csv
```

Optional preprocessing knobs mirror the older `prepare_batch` behavior:

```bash
python neural_network_experiments/train_network.py \
  --model deeponet \
  --framework Framework2 \
  --pde Conservation_law \
  --x-num-model 64 \
  --output-step 2 \
  --normalization per_sample_input
```

- `--x-num-model 64`: uniformly subsample the spatial grid to 64 points.
- `--output-step 2`: train on every other output time step.
- `--output-start-index k`: choose the first output time index. By default this matches the input time index.
- `--normalization global`: normalize with training-set statistics.
- `--normalization per_sample_input`: normalize each sample using the mean/std of its input initial condition, like the older notebook.
- `--normalization none`: no solution normalization.

For an interactive notebook with training curves, prediction/error plots, and a benchmark table, open:

```text
neural_network_experiments/train_and_visualize.ipynb
```

To run on Framework1 datasets with sample-mode logic:

```bash
python neural_network_experiments/train_network.py \
  --model fno \
  --framework Framework1 \
  --pde Conservation_law \
  --epochs 5
```

For Framework1 `Parametric_Wave`, use that exact folder name. For Framework2, use `Param_wave`.
