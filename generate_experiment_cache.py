from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF

from experiment_runner import (
    ExperimentConfig,
    ProductPDEKernel,
    load_data,
    prepare_arrays,
    preprocess,
    relative_errors,
)
from experiment_configs import PDE_CONFIGS, PDE_ORDER, cache_path, slug


DEFAULT_EXPERIMENTS = [
    "framework1_operator",
    "framework2_product",
    "framework2_on_framework1_product",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full prediction caches for paper qualitative figures.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("experiment_cache"))
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--pdes", nargs="+", default=PDE_ORDER)
    parser.add_argument("--splits", nargs="+", default=["test", "ood"], choices=["test", "ood"])
    parser.add_argument(
        "--n-pca",
        type=int,
        default=None,
        help="Optional override for all PCA component counts. By default, paper/table dimensions are used per PDE.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--ood-glob", default="ood*.h5")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def pca_config_kwargs(
    use_pca: bool,
    use_x_pca: bool,
    use_coef_pca: bool,
    n_x_pca: int,
    n_coef_pca: int,
    n_y_pca: int,
) -> dict[str, Any]:
    return {
        "use_pca": use_pca,
        "use_x_pca": use_x_pca if use_pca else False,
        "use_coef_pca": use_coef_pca if use_pca else False,
        "use_y_pca": True,
        "y_pca_by_time": True,
        "n_x_pca": n_x_pca,
        "n_coef_pca": n_coef_pca,
        "n_y_pca": n_y_pca,
    }


def framework1_config(pde_key: str, use_pca: bool, args: argparse.Namespace) -> ExperimentConfig:
    pde = PDE_CONFIGS[pde_key]
    n_x_pca = args.n_pca if args.n_pca is not None else pde.ov_pca_x_dim
    n_y_pca = args.n_pca if args.n_pca is not None else pde.ov_pca_y_dim
    return ExperimentConfig(
        task_mode="parameter_set",
        pde_name=pde.framework1_name,
        base_dir="Framework1",
        results_dir="Framework1",
        use_ood=False,
        train_fraction=args.train_fraction,
        train_size=None,
        framework1_rbf_length_scale=pde.ov_rbf_gamma,
        framework1_matern_length_scale=pde.ov_matern_gamma,
        framework1_matern_nu=pde.ov_matern_nu,
        methods=("method_matern", "method_rbf"),
        save_csv=False,
        **pca_config_kwargs(use_pca, pde.ov_pca_alpha, pde.ov_pca_alpha, n_x_pca, n_x_pca, n_y_pca),
    )


def framework2_config(
    pde_key: str,
    use_pca: bool,
    args: argparse.Namespace,
    on_framework1_data: bool = False,
) -> ExperimentConfig:
    pde = PDE_CONFIGS[pde_key]
    n_x_pca = args.n_pca if args.n_pca is not None else pde.ps_pca_x_dim
    n_coef_pca = args.n_pca if args.n_pca is not None else pde.ps_pca_coef_dim
    n_y_pca = args.n_pca if args.n_pca is not None else pde.ps_pca_y_dim
    base_dir = "Framework1" if on_framework1_data else "Framework2"
    pde_name = pde.framework1_name if on_framework1_data else pde.framework2_name
    train_size = None if on_framework1_data else args.train_size
    if on_framework1_data:
        rbf_coef = pde.ov_ps_kw_rbf_gamma
        matern_coef = pde.ov_ps_kw_matern_gamma
        matern_coef_nu = pde.ov_ps_kw_matern_nu
        rbf_state = pde.ov_ps_ku_rbf_gamma
        matern_state = pde.ov_ps_ku_matern_gamma
        matern_state_nu = pde.ov_ps_ku_matern_nu
        include_alpha_pca = True
    else:
        rbf_coef = pde.ps_kw_rbf_gamma
        matern_coef = pde.ps_kw_matern_gamma
        matern_coef_nu = pde.ps_kw_matern_nu
        rbf_state = pde.ps_ku_rbf_gamma
        matern_state = pde.ps_ku_matern_gamma
        matern_state_nu = pde.ps_ku_matern_nu
        include_alpha_pca = True
    return ExperimentConfig(
        task_mode="sample",
        pde_name=pde_name,
        base_dir=base_dir,
        results_dir="resultsFramework2onFramework1" if on_framework1_data else "Framework2",
        use_ood=False,
        train_fraction=args.train_fraction,
        train_size=train_size,
        vanilla_rbf_length_scale=pde.ps_kernel_o_rbf_gamma,
        vanilla_matern_length_scale=pde.ps_kernel_o_matern_gamma,
        vanilla_matern_nu=pde.ps_kernel_o_matern_nu,
        product_rbf_coef_length_scale=rbf_coef,
        product_matern_coef_length_scale=matern_coef,
        product_matern_coef_nu=matern_coef_nu,
        product_rbf_state_length_scale=rbf_state,
        product_matern_state_length_scale=matern_state,
        product_matern_state_nu=matern_state_nu,
        methods=("method_matern", "method_rbf"),
        save_csv=False,
        **pca_config_kwargs(use_pca, True, include_alpha_pca, n_x_pca, n_coef_pca, n_y_pca),
    )


def fit_framework1_models(cfg: ExperimentConfig, arrays: dict[str, Any]) -> dict[str, GaussianProcessRegressor]:
    specs = {
        "matern": Matern(length_scale=cfg.framework1_matern_length_scale, nu=cfg.framework1_matern_nu),
        "rbf": RBF(length_scale=cfg.framework1_rbf_length_scale),
    }
    models = {}
    for key, kernel in specs.items():
        model = GaussianProcessRegressor(kernel, alpha=cfg.alpha, optimizer=None)
        start = perf_counter()
        model.fit(arrays["fw1_X_train_model"], arrays["fw1_Y_train_model"])
        print(f"  fitted Framework1 {key} in {perf_counter() - start:.2f}s")
        models[key] = model
    return models


def fit_framework2_models(cfg: ExperimentConfig, arrays: dict[str, Any]) -> dict[str, GaussianProcessRegressor]:
    specs = {
        "matern": ProductPDEKernel(
            n_u=arrays["X_train_model"].shape[1],
            coef_kernel=Matern(length_scale=cfg.product_matern_coef_length_scale, nu=cfg.product_matern_coef_nu),
            state_kernel=Matern(length_scale=cfg.product_matern_state_length_scale, nu=cfg.product_matern_state_nu),
        ),
        "rbf": ProductPDEKernel(
            n_u=arrays["X_train_model"].shape[1],
            coef_kernel=RBF(length_scale=cfg.product_rbf_coef_length_scale),
            state_kernel=RBF(length_scale=cfg.product_rbf_state_length_scale),
        ),
    }
    X_train = np.hstack([arrays["X_train_model"], arrays["coef_train_model"]])
    models = {}
    for key, kernel in specs.items():
        model = GaussianProcessRegressor(kernel, alpha=cfg.alpha, optimizer=None)
        start = perf_counter()
        model.fit(X_train, arrays["Y_train_model"])
        print(f"  fitted product {key} in {perf_counter() - start:.2f}s")
        models[key] = model
    return models


def std_like_prediction(std: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    std = np.asarray(std)
    prediction = np.asarray(prediction)
    if std.shape == prediction.shape:
        return np.abs(std)
    if std.size == prediction.size:
        return np.abs(std.reshape(prediction.shape))
    if std.ndim == 1 and std.shape[0] == prediction.shape[0]:
        broadcast_shape = (std.shape[0],) + (1,) * (prediction.ndim - 1)
        return np.broadcast_to(np.abs(std).reshape(broadcast_shape), prediction.shape)
    raise ValueError(f"Cannot align std shape {std.shape} with prediction shape {prediction.shape}.")


def load_ood_raw(cfg: ExperimentConfig, path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return handle[cfg.data_key][:], handle[cfg.coeff_key][:]


def framework1_test_split(arrays: dict[str, Any]) -> dict[str, np.ndarray]:
    n_ic = arrays["fw1_Y_test"].shape[1]
    return {
        "input_model": arrays["fw1_X_test_model"],
        "target": arrays["fw1_Y_test"],
        "input_u": arrays["X_test"].reshape(arrays["fw1_Y_test"].shape[0], n_ic, arrays["X_test"].shape[-1]),
        "coef": arrays["fw1_X_test"],
    }


def framework2_test_split(arrays: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "input_model": np.hstack([arrays["X_test_model"], arrays["coef_test_model"]]),
        "target": arrays["Y_test"],
        "input_u": arrays["X_test"],
        "coef": arrays["coef_test"],
    }


def framework1_ood_split(cfg: ExperimentConfig, arrays: dict[str, Any], path: Path) -> dict[str, np.ndarray]:
    dataset_ood, coef_ood = load_ood_raw(cfg, path)
    n_params = dataset_ood.shape[0] // cfg.ics_per_param
    n_samples = n_params * cfg.ics_per_param
    dataset_ood = dataset_ood[:n_samples]
    coef_ood = coef_ood[:n_samples]
    fw1_X = coef_ood[:: cfg.ics_per_param]
    fw1_Y = dataset_ood[:, :, :, cfg.channel_index].reshape(
        n_params, cfg.ics_per_param, dataset_ood.shape[1], dataset_ood.shape[2]
    )
    pca = arrays.get("fw1_x_pca")
    input_model = pca.transform(fw1_X) if pca is not None else fw1_X
    input_u = dataset_ood[:, cfg.input_time_index, :, cfg.channel_index].reshape(
        n_params, cfg.ics_per_param, dataset_ood.shape[2]
    )
    return {
        "input_model": input_model,
        "target": fw1_Y,
        "input_u": input_u,
        "coef": fw1_X,
    }


def framework2_ood_split(cfg: ExperimentConfig, arrays: dict[str, Any], path: Path) -> dict[str, np.ndarray]:
    dataset_ood, coef_ood = load_ood_raw(cfg, path)
    X = dataset_ood[:, cfg.input_time_index, :, cfg.channel_index]
    Y = dataset_ood[:, :, :, cfg.channel_index]
    x_pca = arrays.get("x_pca")
    coef_pca = arrays.get("coef_pca")
    x_model = x_pca.transform(X) if x_pca is not None else X
    coef_model = coef_pca.transform(coef_ood) if coef_pca is not None else coef_ood
    return {
        "input_model": np.hstack([x_model, coef_model]),
        "target": Y,
        "input_u": X,
        "coef": coef_ood,
    }


def predict_framework1(model, arrays, split_data: dict[str, np.ndarray]):
    pred_flat, pred_std = model.predict(split_data["input_model"], return_std=True)
    mean = arrays["fw1_y_pca"].inverse_transform(pred_flat)
    raw_std = arrays["fw1_y_pca"].inverse_std(pred_std)
    pointwise_std = std_like_prediction(raw_std, mean)
    return mean, pointwise_std


def predict_framework2(model, arrays, split_data: dict[str, np.ndarray]):
    pred_flat, pred_std = model.predict(split_data["input_model"], return_std=True)
    mean = arrays["y_pca"].inverse_transform(pred_flat)
    raw_std = arrays["y_pca"].inverse_std(pred_std)
    pointwise_std = std_like_prediction(raw_std, mean)
    return mean, pointwise_std


def save_bundle(
    path: Path,
    metadata: dict[str, Any],
    split_data: dict[str, np.ndarray],
    mean: np.ndarray,
    pointwise_std: np.ndarray,
    overwrite: bool,
    dry_run: bool,
):
    if path.exists() and not overwrite:
        print(f"  exists, skipping {path}")
        return
    errors = relative_errors(mean, split_data["target"], metadata["task_mode"])
    metadata = dict(
        metadata,
        n_eval=int(len(errors)),
        mean_relative_error=float(np.mean(errors)),
        std_relative_error=float(np.std(errors)),
    )
    print(f"  saving {path} ({len(errors)} eval errors, mean={100 * np.mean(errors):.2f}%)")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        target=split_data["target"],
        mean=mean,
        pointwise_std=pointwise_std,
        input_u=split_data["input_u"],
        coef=split_data["coef"],
        per_item_relative_error=errors,
        metadata=np.array(json.dumps(metadata, indent=2, sort_keys=True)),
    )


def ood_paths_for(cfg: ExperimentConfig, args: argparse.Namespace) -> list[Path]:
    base_dir = Path(cfg.base_dir or ("Framework1" if cfg.task_mode == "parameter_set" else "Framework2"))
    pde_dir = base_dir / cfg.pde_name / cfg.data_subdir
    return sorted(pde_dir.glob(args.ood_glob))


def run_one(experiment: str, pde_key: str, use_pca: bool, args: argparse.Namespace):
    if experiment == "framework1_operator":
        cfg = framework1_config(pde_key, use_pca, args)
        fit_models = fit_framework1_models
        get_test = framework1_test_split
        get_ood = lambda arrays, path: framework1_ood_split(cfg, arrays, path)
        predict = predict_framework1
        method_prefix = "kernelmo_ov"
    elif experiment == "framework2_product":
        cfg = framework2_config(pde_key, use_pca, args, on_framework1_data=False)
        fit_models = fit_framework2_models
        get_test = framework2_test_split
        get_ood = lambda arrays, path: framework2_ood_split(cfg, arrays, path)
        predict = predict_framework2
        method_prefix = "kernelmo_ps"
    elif experiment == "framework2_on_framework1_product":
        cfg = framework2_config(pde_key, use_pca, args, on_framework1_data=True)
        fit_models = fit_framework2_models
        get_test = framework2_test_split
        get_ood = lambda arrays, path: framework2_ood_split(cfg, arrays, path)
        predict = predict_framework2
        method_prefix = "kernelmo_ps"
    else:
        raise ValueError(f"Unknown experiment {experiment!r}.")

    pca_tag = "PCA" if use_pca else "no PCA"
    print(f"\n{experiment} | {pde_key} | {pca_tag}")
    arrays = preprocess(cfg, prepare_arrays(cfg, load_data(cfg)))
    models = fit_models(cfg, arrays)

    split_items = []
    if "test" in args.splits:
        split_items.append(("test", None, get_test(arrays)))
    if "ood" in args.splits:
        for path in ood_paths_for(cfg, args):
            split_items.append((f"ood_{slug(path.name)}", path.name, get_ood(arrays, path)))

    for split_key, ood_filename, split_data in split_items:
        for kernel_key, model in models.items():
            method_key = f"{method_prefix}_{kernel_key}"
            if use_pca:
                method_key += "_pca"
            out_path = cache_path(args.cache_dir, experiment, pde_key, split_key, method_key, use_pca)
            if out_path.exists() and not args.overwrite:
                print(f"  exists, skipping {out_path}")
                continue
            mean, pointwise_std = predict(model, arrays, split_data)
            metadata = {
                "experiment": experiment,
                "pde_key": pde_key,
                "pde_name": cfg.pde_name,
                "task_mode": cfg.task_mode,
                "base_dir": cfg.base_dir,
                "split_key": split_key,
                "ood_filename": ood_filename,
                "kernel": "Matern" if kernel_key == "matern" else "RBF",
                "method_key": method_key,
                "use_pca": use_pca,
                "config": asdict(cfg),
            }
            save_bundle(out_path, metadata, split_data, mean, pointwise_std, args.overwrite, args.dry_run)


def main():
    args = parse_args()
    for experiment in args.experiments:
        for pde_key in args.pdes:
            if pde_key not in PDE_CONFIGS:
                raise KeyError(f"Unknown PDE key {pde_key!r}. Choices: {sorted(PDE_CONFIGS)}")
            for use_pca in (False, True):
                run_one(experiment, pde_key, use_pca, args)


if __name__ == "__main__":
    main()
