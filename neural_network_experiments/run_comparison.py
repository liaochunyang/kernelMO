"""Operator-learning comparison driver for kernelMO.

Trains the four neural-operator comparison settings on both frameworks and
every PDE, optionally at several network sizes, then writes a single tidy CSV
with one row per (framework, pde, setting, size, split):

    1. deeponet_nocoef -> DeepONet (no coefficient input)
    2. deeponet        -> DeepONet (coefficients concatenated to the IC)
    3. mionet          -> MIONet (separate branch nets for IC and coefficients)
    4. mno             -> MNO (tensorized leaf/branch/trunk model)

The train/test split matches the kernel-method experiments exactly (first-N,
no random shuffle), so the test indices line up with the kernel results:

    Framework2: first 10000 samples train, remaining 4000 test.
    Framework1: first 80% train, last 20% test.

Each row records `mean_relative_error`, `train_time_seconds`,
`predict_time_seconds`, and `n_params`, so the notebook can compare accuracy,
model size, and training/inference cost side by side.

Out-of-distribution rows are added automatically for any PDE that ships an
``ood.h5`` next to its ``solutions.h5``.

Examples:

    # default: medium size, all settings/frameworks/PDEs
    python neural_network_experiments/run_comparison.py --epochs 50 --batch-size 64

    # sweep three network sizes
    python neural_network_experiments/run_comparison.py --sizes small medium large
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train_network import parse_args as build_train_args, run_single

# Model key -> human-readable comparison setting label.
SETTINGS = {
    "deeponet_nocoef": "DeepONet (no coeff)",
    "deeponet": "DeepONet (coeff concat)",
    "mionet": "MIONet",
    "mno": "MNO",
}

# Network-size presets. `num_trunk`/`num_branch` drive the DeepONet/MNO branch
# output; `latent` drives MIONet; `hidden_dim` is the MLP width for all of them.
SIZE_PRESETS = {
    "small": {"num_trunk": 50, "num_branch": 50, "latent_dim": 64, "hidden_dim": 100},
    "medium": {"num_trunk": 100, "num_branch": 100, "latent_dim": 128, "hidden_dim": 200},
    "large": {"num_trunk": 150, "num_branch": 150, "latent_dim": 256, "hidden_dim": 400},
}

# PDE folder names differ between the two frameworks (Parametric_Wave vs Param_wave).
FRAMEWORK_PDES = {
    "Framework2": [
        "Conservation_law",
        "DiffReacAdv",
        "Nonlinear_Klein_Gordon",
        "Param_DiffReac",
        "Param_wave",
    ],
    "Framework1": [
        "Conservation_law",
        "DiffReacAdv",
        "Nonlinear_Klein_Gordon",
        "Param_DiffReac",
        "Parametric_Wave",
    ],
}


def split_argv(framework: str) -> list[str]:
    """Train/test split flags matching the kernel-method convention."""
    if framework == "Framework2":
        return ["--train-size", "10000"]
    # Framework1: first 80% train, last 20% test (train-size<=0 -> use fraction).
    return ["--train-size", "-1", "--train-fraction", "0.8"]


def build_run_argv(model: str, framework: str, pde: str, size_cfg: dict, args) -> list[str]:
    argv = [
        "--model", model,
        "--framework", framework,
        "--pde", pde,
        "--data-root", args.data_root,
        "--data-subdir", args.data_subdir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--normalization", args.normalization,
        "--trunk-batch-size", str(args.trunk_batch_size),
        "--num-trunk", str(size_cfg["num_trunk"]),
        "--num-branch", str(size_cfg["num_branch"]),
        "--num-leaf", str(args.num_leaf),
        "--latent-dim", str(size_cfg["latent_dim"]),
        "--hidden-dim", str(size_cfg["hidden_dim"]),
        "--seed", str(args.seed),
        "--log-every", str(args.log_every),
    ]
    if args.device:
        argv += ["--device", args.device]
    if args.no_ood:
        argv += ["--no-ood"]
    if args.ood_filenames:
        argv += ["--ood-filenames", *args.ood_filenames]
    argv += split_argv(framework)
    return argv


def main():
    args = parse_args()
    frameworks = args.frameworks or list(FRAMEWORK_PDES)
    models = args.models or list(SETTINGS)
    sizes = args.sizes or ["medium"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for framework in frameworks:
        pdes = args.pdes or FRAMEWORK_PDES[framework]
        for pde in pdes:
            for model in models:
                label = SETTINGS.get(model, model)
                for size_name in sizes:
                    size_cfg = SIZE_PRESETS[size_name]
                    print(f"\n=== {framework} / {pde} / {label} / {size_name} ===")
                    run_args = build_train_args(build_run_argv(model, framework, pde, size_cfg, args))
                    if args.checkpoint_dir:
                        run_args.checkpoint = str(
                            Path(args.checkpoint_dir) / f"{framework}_{pde}_{model}_{size_name}.pt"
                        )
                    try:
                        df = run_single(run_args)
                    except FileNotFoundError as exc:
                        print(f"  skipped ({exc})")
                        continue
                    df.insert(3, "setting", label)
                    df.insert(4, "size", size_name)
                    frames.append(df)
                    pd.concat(frames, ignore_index=True).to_csv(output, index=False)

    if not frames:
        print("No runs completed (missing data files?).")
        return
    result = pd.concat(frames, ignore_index=True)
    print(f"\nSaved {len(result)} rows to {output}")
    test = result[result["split"] == "test"]
    if not test.empty:
        index = ["framework", "pde"] + (["size"] if len(sizes) > 1 else [])
        pivot = test.pivot_table(index=index, columns="setting", values="mean_relative_error")
        print("\nTest mean relative error:\n")
        print(pivot.to_string(float_format=lambda v: f"{v:.3e}"))


def parse_args():
    parser = argparse.ArgumentParser(description="Run the operator-learning comparison across both frameworks.")
    parser.add_argument("--frameworks", nargs="*", choices=list(FRAMEWORK_PDES), default=None,
                        help="Subset of frameworks to run (default: both).")
    parser.add_argument("--pdes", nargs="*", default=None,
                        help="Subset of PDE folder names (default: all PDEs for each framework).")
    parser.add_argument("--models", nargs="*", choices=list(SETTINGS), default=None,
                        help="Subset of comparison settings (default: all four).")
    parser.add_argument("--sizes", nargs="*", choices=list(SIZE_PRESETS), default=None,
                        help="Network-size presets to sweep (default: medium only).")
    parser.add_argument("--data-root", default="/home/shared/dataset/KernelMOL",
                        help="Root holding Framework{1,2}/<pde>/dataset_simple/*.h5.")
    parser.add_argument("--data-subdir", default="dataset_simple")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--normalization", choices=["global", "per_sample_input", "none"], default="global")
    parser.add_argument("--trunk-batch-size", type=int, default=2048)
    parser.add_argument("--num-leaf", type=int, default=2)
    parser.add_argument("--no-ood", action="store_true")
    parser.add_argument("--ood-filenames", nargs="*", default=["auto"],
                        help="OOD h5 basenames to evaluate per PDE. 'auto' (default) discovers every ood*.h5.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default=None, help="If set, save each trained model here.")
    parser.add_argument("--output", default="neural_network_experiments/results_comparison.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main()
