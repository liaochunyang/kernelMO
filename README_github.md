# Kernel Multiple Operator Learning Experiments

This repository contains the code and datasets used to run kernel multiple-operator learning experiments for several PDE benchmarks. The main workflows are:

1. run the experiments from the HDF5 datasets,
2. cache model predictions and uncertainty estimates,
3. summarize the cached results into CSV tables.

## Repository Structure

The main reusable files are:

```text
mainComplete_all_ood.ipynb
experiment_runner.py
experiment_configs.py
generate_experiment_cache.py
summarize_experiment_cache.py
view_experiment_summary.ipynb
Framework1/
Framework2/
```

The roles are:

```text
experiment_runner.py
```

Core experiment logic: data loading, train/test/OOD splitting, PCA preprocessing, kernel fitting, prediction, uncertainty extraction, and relative error computation.

```text
experiment_configs.py
```

PDE names, framework names, hyperparameters, PCA dimensions, and cache path helpers.

```text
generate_experiment_cache.py
```

Batch script for running the configured experiments and saving cached predictions as `.npz` files.

```text
summarize_experiment_cache.py
```

Reads cached `.npz` files and writes a CSV summary of mean/std relative errors.

```text
view_experiment_summary.ipynb
```

Notebook for opening and displaying the summary CSV with all rows visible.

## Data

The experiment scripts expect datasets in the following layout:

```text
Framework1/<PDE>/dataset_simple/solutions.h5
Framework1/<PDE>/dataset_simple/ood*.h5

Framework2/<PDE>/dataset_simple/solutions.h5
Framework2/<PDE>/dataset_simple/ood*.h5
```

Each HDF5 file should contain:

```text
data
coeffs
```

where `data` stores the PDE solution trajectories and `coeffs` stores the corresponding PDE parameters or coefficient functions.

The `data_generation.ipynb` files inside the PDE folders generate the in-distribution datasets and some OOD datasets. Some named OOD datasets may need to be provided directly or regenerated with additional generation code.

## Generate Cached Predictions

To run all configured experiments and save prediction caches:

```bash
python generate_experiment_cache.py \
  --experiments framework1_operator framework2_product framework2_on_framework1_product \
  --pdes conservation diffreacadv klein_gordon param_diffreac param_wave \
  --splits test ood \
  --overwrite
```

By default, cached files are written to:

```text
experiment_cache/
```

Each cached `.npz` file contains:

```text
target
mean
pointwise_std
input_u
coef
per_item_relative_error
metadata
```

## Summarize Cached Results

After caches exist, create a CSV summary with:

```bash
python summarize_experiment_cache.py
```

This writes:

```text
experiment_cache/cache_mean_summary.csv
```

To also save one row per evaluated sample:

```bash
python summarize_experiment_cache.py \
  --details-output experiment_cache/cache_item_errors.csv
```

## View Results

Open:

```text
view_experiment_summary.ipynb
```

This notebook loads:

```text
experiment_cache/cache_mean_summary.csv
```

and displays the full table, a compact view, a pivot table, and optional filtered views.

