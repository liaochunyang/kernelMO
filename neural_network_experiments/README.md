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
- `deeponet`: a tensorized DeepONet matching the implementation in `main.ipynb`; the branch input is `[coefficients, u0]`, and the trunk input is `(t, x)`.
- `mno`: the MNO-style tensorized model from `main.ipynb`; it uses separate trunk `(t, x)`, branch `u0`, and leaf `coefficients` networks.

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
