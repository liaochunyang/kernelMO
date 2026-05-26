from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Matern, RBF


@dataclass(frozen=True)
class ExperimentConfig:
    task_mode: str = "sample"  # "sample" for Framework2, "parameter_set" for Framework1
    pde_name: str = "Conservation_law"
    base_dir: str | None = None
    results_dir: str | None = None
    data_subdir: str = "dataset_simple"
    data_filename: str = "solutions.h5"
    ood_filename: str = "ood.h5"
    use_ood: bool = True
    data_key: str = "data"
    coeff_key: str = "coeffs"
    input_time_index: int = 0
    channel_index: int = 0
    ics_per_param: int = 20
    train_fraction: float = 0.8
    train_size: int | None = None
    validation_fraction: float = 0.0  # fraction of the original test split reserved for validation
    alpha: float = 1e-10
    use_pca: bool = False
    use_x_pca: bool = True
    use_coef_pca: bool = True
    use_y_pca: bool = True
    y_pca_by_time: bool = True
    n_x_pca: int = 5
    n_coef_pca: int = 4
    n_y_pca: int = 5
    vanilla_rbf_length_scale: float = 100.0
    vanilla_matern_length_scale: float = 100.0
    vanilla_matern_nu: float = 1.5
    framework1_rbf_length_scale: float = 0.1
    framework1_matern_length_scale: float = 1.0
    framework1_matern_nu: float = 2.5
    product_rbf_coef_length_scale: float = 1.0
    product_rbf_state_length_scale: float = 100.0
    product_matern_coef_length_scale: float = 1.0
    product_matern_coef_nu: float = 2.5
    product_matern_state_length_scale: float = 100.0
    product_matern_state_nu: float = 1.5
    methods: tuple[str, ...] = ("vanilla_rbf", "vanilla_matern", "method_matern", "method_rbf")
    save_csv: bool = True
    results_filename: str | None = None
    sweep_id: str | None = None


def default_base_dir(task_mode: str) -> Path:
    if task_mode == "parameter_set":
        return Path("Framework1")
    if task_mode == "sample":
        return Path("Framework2")
    raise ValueError("task_mode must be 'parameter_set' or 'sample'")


def relative_errors(pred: np.ndarray, target: np.ndarray, task_mode: str | None = None) -> np.ndarray:
    pred = np.asarray(pred)
    target = np.asarray(target)
    if task_mode == "parameter_set" and pred.ndim == 4:
        pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
        target = target.reshape(-1, target.shape[-2], target.shape[-1])
    axes = tuple(range(1, pred.ndim))
    numerator = np.sqrt(np.sum((pred - target) ** 2, axis=axes))
    denominator = np.sqrt(np.sum(target**2, axis=axes))
    return numerator / denominator


def sample_uncertainty(std: np.ndarray) -> np.ndarray:
    std = np.asarray(std)
    if std.ndim == 1:
        return std
    return np.sqrt(np.mean(std**2, axis=tuple(range(1, std.ndim))))


class ProductPDEKernel(Kernel):
    def __init__(self, n_u: int, coef_kernel: Kernel | None = None, state_kernel: Kernel | None = None):
        self.n_u = n_u
        self.coef_kernel = coef_kernel
        self.state_kernel = state_kernel

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise ValueError("Gradient is not implemented. Use optimizer=None.")
        X_u = X[:, : self.n_u]
        X_c = X[:, self.n_u :]
        if Y is None:
            Y_u = None
            Y_c = None
        else:
            Y_u = Y[:, : self.n_u]
            Y_c = Y[:, self.n_u :]
        return self.state_kernel(X_u, Y_u) * self.coef_kernel(X_c, Y_c)

    def diag(self, X):
        X_u = X[:, : self.n_u]
        X_c = X[:, self.n_u :]
        return self.state_kernel.diag(X_u) * self.coef_kernel.diag(X_c)

    def is_stationary(self):
        return self.state_kernel.is_stationary() and self.coef_kernel.is_stationary()


class OutputPCA:
    def __init__(self, use_pca: bool, use_y_pca: bool, by_time: bool, n_components: int):
        self.use_pca = use_pca
        self.use_y_pca = use_y_pca
        self.by_time = by_time
        self.n_components = n_components
        self.shape_: tuple[int, ...] | None = None
        self.pca_: PCA | None = None
        self.pcas_: list[PCA] = []
        self.slices_: list[slice] = []
        self.per_time_shape_: tuple[int, ...] | None = None

    @staticmethod
    def _by_time(Y: np.ndarray) -> np.ndarray:
        return np.moveaxis(Y, -2, 1)

    @staticmethod
    def _from_time(Y: np.ndarray) -> np.ndarray:
        return np.moveaxis(Y, 1, -2)

    def fit_transform(self, Y: np.ndarray) -> np.ndarray:
        self.shape_ = Y.shape[1:]
        Y_flat = Y.reshape(Y.shape[0], -1)
        if not (self.use_pca and self.use_y_pca):
            return Y_flat
        if not self.by_time:
            n_components = min(self.n_components, Y_flat.shape[0], Y_flat.shape[1])
            self.pca_ = PCA(n_components=n_components)
            return self.pca_.fit_transform(Y_flat)

        parts = []
        start = 0
        Y_time = self._by_time(Y)
        self.per_time_shape_ = Y_time.shape[2:]
        for t in range(Y_time.shape[1]):
            Y_t = Y_time[:, t].reshape(Y.shape[0], -1)
            n_components = min(self.n_components, Y_t.shape[0], Y_t.shape[1])
            pca = PCA(n_components=n_components)
            part = pca.fit_transform(Y_t)
            self.pcas_.append(pca)
            parts.append(part)
            stop = start + n_components
            self.slices_.append(slice(start, stop))
            start = stop
        return np.hstack(parts)

    def inverse_transform(self, prediction: np.ndarray) -> np.ndarray:
        prediction = np.asarray(prediction)
        if not (self.use_pca and self.use_y_pca):
            return prediction.reshape((-1,) + self.shape_)
        if self.pca_ is not None:
            return self.pca_.inverse_transform(prediction).reshape((-1,) + self.shape_)

        decoded = []
        for pca, sl in zip(self.pcas_, self.slices_):
            decoded_t = pca.inverse_transform(prediction[:, sl]).reshape((prediction.shape[0],) + self.per_time_shape_)
            decoded.append(decoded_t)
        return self._from_time(np.stack(decoded, axis=1))

    def inverse_std(self, std: np.ndarray) -> np.ndarray:
        std = np.asarray(std)
        if std.ndim == 1 or not (self.use_pca and self.use_y_pca):
            return std
        if self.pca_ is not None:
            var = (std**2) @ (self.pca_.components_**2)
            return np.sqrt(np.maximum(var, 0.0)).reshape((-1,) + self.shape_)

        decoded = []
        for pca, sl in zip(self.pcas_, self.slices_):
            var_t = (std[:, sl] ** 2) @ (pca.components_**2)
            decoded_t = np.sqrt(np.maximum(var_t, 0.0)).reshape((std.shape[0],) + self.per_time_shape_)
            decoded.append(decoded_t)
        return self._from_time(np.stack(decoded, axis=1))


class Framework1OutputPCA:
    def __init__(self, use_pca: bool, use_y_pca: bool, by_time: bool, n_components: int):
        self.use_pca = use_pca
        self.use_y_pca = use_y_pca
        self.by_time = by_time
        self.n_components = n_components
        self.shape_: tuple[int, ...] | None = None
        self.pca_: PCA | None = None
        self.pcas_: list[PCA] = []
        self.slices_: list[tuple[slice, int]] = []
        self.n_ic_: int | None = None
        self.n_x_: int | None = None

    def fit_transform(self, Y: np.ndarray) -> np.ndarray:
        self.shape_ = Y.shape[1:]
        Y_flat = Y.reshape(Y.shape[0], -1)
        if not (self.use_pca and self.use_y_pca):
            return Y_flat
        if not self.by_time:
            n_components = min(self.n_components, Y_flat.shape[0], Y_flat.shape[1])
            self.pca_ = PCA(n_components=n_components)
            return self.pca_.fit_transform(Y_flat)

        n_params, n_ic, n_t, n_x = Y.shape
        self.n_ic_ = n_ic
        self.n_x_ = n_x
        parts = []
        start = 0
        for t in range(n_t):
            Y_t = Y[:, :, t, :].reshape(n_params * n_ic, n_x)
            n_components = min(self.n_components, Y_t.shape[0], Y_t.shape[1])
            pca = PCA(n_components=n_components)
            part = pca.fit_transform(Y_t).reshape(n_params, n_ic * n_components)
            self.pcas_.append(pca)
            parts.append(part)
            stop = start + n_ic * n_components
            self.slices_.append((slice(start, stop), n_components))
            start = stop
        return np.hstack(parts)

    def inverse_transform(self, prediction: np.ndarray) -> np.ndarray:
        prediction = np.asarray(prediction)
        if not (self.use_pca and self.use_y_pca):
            return prediction.reshape((-1,) + self.shape_)
        if self.pca_ is not None:
            return self.pca_.inverse_transform(prediction).reshape((-1,) + self.shape_)

        decoded = []
        for pca, (sl, n_components) in zip(self.pcas_, self.slices_):
            coeffs = prediction[:, sl].reshape(prediction.shape[0] * self.n_ic_, n_components)
            decoded_t = pca.inverse_transform(coeffs).reshape(prediction.shape[0], self.n_ic_, self.n_x_)
            decoded.append(decoded_t)
        return np.stack(decoded, axis=2)

    def inverse_std(self, std: np.ndarray) -> np.ndarray:
        std = np.asarray(std)
        if std.ndim == 1 or not (self.use_pca and self.use_y_pca):
            return std
        if self.pca_ is not None:
            var = (std**2) @ (self.pca_.components_**2)
            return np.sqrt(np.maximum(var, 0.0)).reshape((-1,) + self.shape_)

        decoded = []
        for pca, (sl, n_components) in zip(self.pcas_, self.slices_):
            std_t = std[:, sl].reshape(std.shape[0] * self.n_ic_, n_components)
            var_t = (std_t**2) @ (pca.components_**2)
            decoded_t = np.sqrt(np.maximum(var_t, 0.0)).reshape(std.shape[0], self.n_ic_, self.n_x_)
            decoded.append(decoded_t)
        return np.stack(decoded, axis=2)


def _fit_pca(train, test, ood, enabled: bool, n_components: int):
    if not enabled:
        return train, test, ood, None
    n_components = min(n_components, train.shape[0], train.shape[1])
    pca = PCA(n_components=n_components)
    train_model = pca.fit_transform(train)
    test_model = pca.transform(test)
    ood_model = pca.transform(ood) if ood is not None else None
    return train_model, test_model, ood_model, pca


def _add_result(rows: list[dict[str, Any]], cfg: ExperimentConfig, split: str, method: str, kernel: str, errors, **params):
    row = {
        "sweep_id": cfg.sweep_id,
        "pde": cfg.pde_name,
        "task_mode": cfg.task_mode,
        "method": method,
        "kernel": kernel,
        "split": split,
        "n_eval": len(errors),
        "mean_relative_error": float(np.mean(errors)),
        "std_relative_error": float(np.std(errors)),
        "pca": cfg.use_pca,
    }
    row.update(params)
    rows.append(row)


def _paths(cfg: ExperimentConfig):
    base_dir = Path(cfg.base_dir) if cfg.base_dir is not None else default_base_dir(cfg.task_mode)
    results_dir = Path(cfg.results_dir) if cfg.results_dir is not None else base_dir
    pde_dir = base_dir / cfg.pde_name
    return base_dir, results_dir, pde_dir / cfg.data_subdir / cfg.data_filename, pde_dir / cfg.data_subdir / cfg.ood_filename


def load_data(cfg: ExperimentConfig) -> dict[str, Any]:
    _, _, data_path, ood_path = _paths(cfg)
    with h5py.File(data_path, "r") as f:
        dataset = f[cfg.data_key][:]
        coef = f[cfg.coeff_key][:]

    data: dict[str, Any] = {"dataset": dataset, "coef": coef, "has_ood": False}
    if cfg.use_ood and ood_path.exists():
        with h5py.File(ood_path, "r") as f:
            data["dataset_ood"] = f[cfg.data_key][:]
            data["coef_ood"] = f[cfg.coeff_key][:]
            data["has_ood"] = True
    return data


def prepare_arrays(cfg: ExperimentConfig, data: dict[str, Any]) -> dict[str, Any]:
    dataset = data["dataset"]
    coef = data["coef"]
    has_ood = data["has_ood"]
    arrays: dict[str, Any] = {"has_ood": has_ood}

    if cfg.task_mode == "parameter_set":
        n_params = dataset.shape[0] // cfg.ics_per_param
        n_samples = n_params * cfg.ics_per_param
        dataset = dataset[:n_samples]
        coef = coef[:n_samples]
        train_params = cfg.train_size if cfg.train_size is not None else int(cfg.train_fraction * n_params)
        train_samples = train_params * cfg.ics_per_param
        coef_by_param = coef[:: cfg.ics_per_param]
        Y_by_param = dataset[:, :, :, cfg.channel_index].reshape(
            n_params, cfg.ics_per_param, dataset.shape[1], dataset.shape[2]
        )

        arrays.update(
            X_train=dataset[:train_samples, cfg.input_time_index, :, cfg.channel_index],
            Y_train=dataset[:train_samples, :, :, cfg.channel_index],
            X_test=dataset[train_samples:, cfg.input_time_index, :, cfg.channel_index],
            Y_test=dataset[train_samples:, :, :, cfg.channel_index],
            coef_train=coef[:train_samples],
            coef_test=coef[train_samples:],
            fw1_X_train=coef_by_param[:train_params],
            fw1_Y_train=Y_by_param[:train_params],
            fw1_X_test=coef_by_param[train_params:],
            fw1_Y_test=Y_by_param[train_params:],
        )
        if has_ood:
            dataset_ood = data["dataset_ood"]
            coef_ood = data["coef_ood"]
            n_ood_params = dataset_ood.shape[0] // cfg.ics_per_param
            n_ood_samples = n_ood_params * cfg.ics_per_param
            dataset_ood = dataset_ood[:n_ood_samples]
            coef_ood = coef_ood[:n_ood_samples]
            arrays.update(
                X_ood=dataset_ood[:, cfg.input_time_index, :, cfg.channel_index],
                Y_ood=dataset_ood[:, :, :, cfg.channel_index],
                coef_ood=coef_ood,
                fw1_X_ood=coef_ood[:: cfg.ics_per_param],
                fw1_Y_ood=dataset_ood[:, :, :, cfg.channel_index].reshape(
                    n_ood_params, cfg.ics_per_param, dataset_ood.shape[1], dataset_ood.shape[2]
                ),
            )
    elif cfg.task_mode == "sample":
        train_size = cfg.train_size if cfg.train_size is not None else int(cfg.train_fraction * dataset.shape[0])
        arrays.update(
            X_train=dataset[:train_size, cfg.input_time_index, :, cfg.channel_index],
            Y_train=dataset[:train_size, :, :, cfg.channel_index],
            X_test=dataset[train_size:, cfg.input_time_index, :, cfg.channel_index],
            Y_test=dataset[train_size:, :, :, cfg.channel_index],
            coef_train=coef[:train_size],
            coef_test=coef[train_size:],
        )
        if has_ood:
            dataset_ood = data["dataset_ood"]
            coef_ood = data["coef_ood"]
            arrays.update(
                X_ood=dataset_ood[:, cfg.input_time_index, :, cfg.channel_index],
                Y_ood=dataset_ood[:, :, :, cfg.channel_index],
                coef_ood=coef_ood,
            )
    else:
        raise ValueError("task_mode must be 'parameter_set' or 'sample'")

    return split_validation(arrays, cfg)


def _split_count(n_items: int, fraction: float) -> int:
    if fraction <= 0:
        return 0
    if fraction >= 1:
        raise ValueError("validation_fraction must be smaller than 1.")
    count = int(round(fraction * n_items))
    return min(max(count, 1), n_items - 1)


def split_validation(arrays: dict[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
    arrays = dict(arrays)
    arrays["has_validation"] = False
    if cfg.validation_fraction <= 0:
        return arrays

    n_val = _split_count(arrays["X_test"].shape[0], cfg.validation_fraction)
    arrays["X_val"] = arrays["X_test"][:n_val]
    arrays["Y_val"] = arrays["Y_test"][:n_val]
    arrays["coef_val"] = arrays["coef_test"][:n_val]
    arrays["X_test"] = arrays["X_test"][n_val:]
    arrays["Y_test"] = arrays["Y_test"][n_val:]
    arrays["coef_test"] = arrays["coef_test"][n_val:]

    if cfg.task_mode == "parameter_set":
        n_fw1_val = _split_count(arrays["fw1_X_test"].shape[0], cfg.validation_fraction)
        arrays["fw1_X_val"] = arrays["fw1_X_test"][:n_fw1_val]
        arrays["fw1_Y_val"] = arrays["fw1_Y_test"][:n_fw1_val]
        arrays["fw1_X_test"] = arrays["fw1_X_test"][n_fw1_val:]
        arrays["fw1_Y_test"] = arrays["fw1_Y_test"][n_fw1_val:]

    arrays["has_validation"] = True
    return arrays


def preprocess(cfg: ExperimentConfig, arrays: dict[str, Any]) -> dict[str, Any]:
    out = dict(arrays)
    has_ood = arrays["has_ood"]
    out["X_train_model"], out["X_test_model"], out["X_ood_model"], out["x_pca"] = _fit_pca(
        arrays["X_train"], arrays["X_test"], arrays.get("X_ood"), cfg.use_pca and cfg.use_x_pca, cfg.n_x_pca
    )
    out["X_val_model"] = out["x_pca"].transform(arrays["X_val"]) if arrays.get("has_validation") and out["x_pca"] is not None else arrays.get("X_val")
    out["coef_train_model"], out["coef_test_model"], out["coef_ood_model"], out["coef_pca"] = _fit_pca(
        arrays["coef_train"],
        arrays["coef_test"],
        arrays.get("coef_ood"),
        cfg.use_pca and cfg.use_coef_pca and cfg.task_mode == "sample",
        cfg.n_coef_pca,
    )
    out["coef_val_model"] = (
        out["coef_pca"].transform(arrays["coef_val"])
        if arrays.get("has_validation") and out["coef_pca"] is not None
        else arrays.get("coef_val")
    )
    y_pca = OutputPCA(cfg.use_pca, cfg.use_y_pca, cfg.y_pca_by_time, cfg.n_y_pca)
    out["Y_train_model"] = y_pca.fit_transform(arrays["Y_train"])
    out["y_pca"] = y_pca

    if cfg.task_mode == "parameter_set":
        out["fw1_X_train_model"], out["fw1_X_test_model"], out["fw1_X_ood_model"], out["fw1_x_pca"] = _fit_pca(
            arrays["fw1_X_train"],
            arrays["fw1_X_test"],
            arrays.get("fw1_X_ood"),
            cfg.use_pca and cfg.use_x_pca,
            cfg.n_x_pca,
        )
        out["fw1_X_val_model"] = (
            out["fw1_x_pca"].transform(arrays["fw1_X_val"])
            if arrays.get("has_validation") and out["fw1_x_pca"] is not None
            else arrays.get("fw1_X_val")
        )
        fw1_y_pca = Framework1OutputPCA(cfg.use_pca, cfg.use_y_pca, cfg.y_pca_by_time, cfg.n_y_pca)
        out["fw1_Y_train_model"] = fw1_y_pca.fit_transform(arrays["fw1_Y_train"])
        out["fw1_y_pca"] = fw1_y_pca
    return out


def run_vanilla(cfg: ExperimentConfig, arrays: dict[str, Any], rows: list[dict[str, Any]], kernel_name: str):
    if kernel_name == "RBF":
        kernel = RBF(length_scale=cfg.vanilla_rbf_length_scale)
        params = {"length_scale": cfg.vanilla_rbf_length_scale}
    else:
        kernel = Matern(length_scale=cfg.vanilla_matern_length_scale, nu=cfg.vanilla_matern_nu)
        params = {"length_scale": cfg.vanilla_matern_length_scale, "nu": cfg.vanilla_matern_nu}

    model = GaussianProcessRegressor(kernel, alpha=cfg.alpha, optimizer=None)
    start = perf_counter()
    model.fit(arrays["X_train_model"], arrays["Y_train_model"])
    train_time = perf_counter() - start
    start = perf_counter()
    pred_flat, pred_std = model.predict(arrays["X_test_model"], return_std=True)
    predict_time = perf_counter() - start
    pred = arrays["y_pca"].inverse_transform(pred_flat)
    errors = relative_errors(pred, arrays["Y_test"], cfg.task_mode)
    _add_result(
        rows,
        cfg,
        "test",
        "vanilla",
        kernel_name,
        errors,
        n_train=arrays["X_train"].shape[0],
        train_time_seconds=train_time,
        predict_time_seconds=predict_time,
        alpha=cfg.alpha,
        **params,
    )
    if arrays.get("has_validation"):
        start = perf_counter()
        pred_flat, pred_std = model.predict(arrays["X_val_model"], return_std=True)
        predict_time = perf_counter() - start
        pred = arrays["y_pca"].inverse_transform(pred_flat)
        errors = relative_errors(pred, arrays["Y_val"], cfg.task_mode)
        _add_result(
            rows,
            cfg,
            "val",
            "vanilla",
            kernel_name,
            errors,
            n_train=arrays["X_train"].shape[0],
            train_time_seconds=train_time,
            predict_time_seconds=predict_time,
            alpha=cfg.alpha,
            **params,
        )
    if arrays["has_ood"]:
        start = perf_counter()
        pred_flat, pred_std = model.predict(arrays["X_ood_model"], return_std=True)
        predict_time = perf_counter() - start
        pred = arrays["y_pca"].inverse_transform(pred_flat)
        errors = relative_errors(pred, arrays["Y_ood"], cfg.task_mode)
        _add_result(
            rows,
            cfg,
            "ood",
            "vanilla",
            kernel_name,
            errors,
            n_train=arrays["X_train"].shape[0],
            train_time_seconds=train_time,
            predict_time_seconds=predict_time,
            alpha=cfg.alpha,
            **params,
        )


def run_framework1_method(cfg: ExperimentConfig, arrays: dict[str, Any], rows: list[dict[str, Any]], kernel_name: str):
    if kernel_name == "RBF":
        kernel = RBF(length_scale=cfg.framework1_rbf_length_scale)
        params = {"length_scale": cfg.framework1_rbf_length_scale}
    else:
        kernel = Matern(length_scale=cfg.framework1_matern_length_scale, nu=cfg.framework1_matern_nu)
        params = {"length_scale": cfg.framework1_matern_length_scale, "nu": cfg.framework1_matern_nu}

    model = GaussianProcessRegressor(kernel, alpha=cfg.alpha, optimizer=None)
    start = perf_counter()
    model.fit(arrays["fw1_X_train_model"], arrays["fw1_Y_train_model"])
    train_time = perf_counter() - start
    start = perf_counter()
    pred_flat, pred_std = model.predict(arrays["fw1_X_test_model"], return_std=True)
    predict_time = perf_counter() - start
    pred = arrays["fw1_y_pca"].inverse_transform(pred_flat)
    errors = relative_errors(pred, arrays["fw1_Y_test"], cfg.task_mode)
    _add_result(
        rows,
        cfg,
        "test",
        "framework1",
        kernel_name,
        errors,
        n_train=arrays["fw1_X_train"].shape[0],
        train_time_seconds=train_time,
        predict_time_seconds=predict_time,
        alpha=cfg.alpha,
        **params,
    )
    if arrays.get("has_validation"):
        start = perf_counter()
        pred_flat, pred_std = model.predict(arrays["fw1_X_val_model"], return_std=True)
        predict_time = perf_counter() - start
        pred = arrays["fw1_y_pca"].inverse_transform(pred_flat)
        errors = relative_errors(pred, arrays["fw1_Y_val"], cfg.task_mode)
        _add_result(
            rows,
            cfg,
            "val",
            "framework1",
            kernel_name,
            errors,
            n_train=arrays["fw1_X_train"].shape[0],
            train_time_seconds=train_time,
            predict_time_seconds=predict_time,
            alpha=cfg.alpha,
            **params,
        )
    if arrays["has_ood"]:
        start = perf_counter()
        pred_flat, pred_std = model.predict(arrays["fw1_X_ood_model"], return_std=True)
        predict_time = perf_counter() - start
        pred = arrays["fw1_y_pca"].inverse_transform(pred_flat)
        errors = relative_errors(pred, arrays["fw1_Y_ood"], cfg.task_mode)
        _add_result(
            rows,
            cfg,
            "ood",
            "framework1",
            kernel_name,
            errors,
            n_train=arrays["fw1_X_train"].shape[0],
            train_time_seconds=train_time,
            predict_time_seconds=predict_time,
            alpha=cfg.alpha,
            **params,
        )


def run_product_method(cfg: ExperimentConfig, arrays: dict[str, Any], rows: list[dict[str, Any]], kernel_name: str):
    if kernel_name == "RBF x RBF":
        kernel = ProductPDEKernel(
            n_u=arrays["X_train_model"].shape[1],
            coef_kernel=RBF(length_scale=cfg.product_rbf_coef_length_scale),
            state_kernel=RBF(length_scale=cfg.product_rbf_state_length_scale),
        )
        params = {
            "coef_length_scale": cfg.product_rbf_coef_length_scale,
            "state_length_scale": cfg.product_rbf_state_length_scale,
        }
    else:
        kernel = ProductPDEKernel(
            n_u=arrays["X_train_model"].shape[1],
            coef_kernel=Matern(length_scale=cfg.product_matern_coef_length_scale, nu=cfg.product_matern_coef_nu),
            state_kernel=Matern(length_scale=cfg.product_matern_state_length_scale, nu=cfg.product_matern_state_nu),
        )
        params = {
            "coef_length_scale": cfg.product_matern_coef_length_scale,
            "coef_nu": cfg.product_matern_coef_nu,
            "state_length_scale": cfg.product_matern_state_length_scale,
            "state_nu": cfg.product_matern_state_nu,
        }

    X_train = np.hstack([arrays["X_train_model"], arrays["coef_train_model"]])
    X_test = np.hstack([arrays["X_test_model"], arrays["coef_test_model"]])
    X_val = (
        np.hstack([arrays["X_val_model"], arrays["coef_val_model"]])
        if arrays.get("has_validation")
        else None
    )
    X_ood = (
        np.hstack([arrays["X_ood_model"], arrays["coef_ood_model"]])
        if arrays["has_ood"]
        else None
    )
    model = GaussianProcessRegressor(kernel, alpha=cfg.alpha, optimizer=None)
    start = perf_counter()
    model.fit(X_train, arrays["Y_train_model"])
    train_time = perf_counter() - start
    start = perf_counter()
    pred_flat, pred_std = model.predict(X_test, return_std=True)
    predict_time = perf_counter() - start
    pred = arrays["y_pca"].inverse_transform(pred_flat)
    errors = relative_errors(pred, arrays["Y_test"], cfg.task_mode)
    _add_result(
        rows,
        cfg,
        "test",
        "product_gpr",
        kernel_name,
        errors,
        n_train=arrays["X_train"].shape[0],
        train_time_seconds=train_time,
        predict_time_seconds=predict_time,
        alpha=cfg.alpha,
        **params,
    )
    if arrays.get("has_validation"):
        start = perf_counter()
        pred_flat, pred_std = model.predict(X_val, return_std=True)
        predict_time = perf_counter() - start
        pred = arrays["y_pca"].inverse_transform(pred_flat)
        errors = relative_errors(pred, arrays["Y_val"], cfg.task_mode)
        _add_result(
            rows,
            cfg,
            "val",
            "product_gpr",
            kernel_name,
            errors,
            n_train=arrays["X_train"].shape[0],
            train_time_seconds=train_time,
            predict_time_seconds=predict_time,
            alpha=cfg.alpha,
            **params,
        )
    if arrays["has_ood"]:
        start = perf_counter()
        pred_flat, pred_std = model.predict(X_ood, return_std=True)
        predict_time = perf_counter() - start
        pred = arrays["y_pca"].inverse_transform(pred_flat)
        errors = relative_errors(pred, arrays["Y_ood"], cfg.task_mode)
        _add_result(
            rows,
            cfg,
            "ood",
            "product_gpr",
            kernel_name,
            errors,
            n_train=arrays["X_train"].shape[0],
            train_time_seconds=train_time,
            predict_time_seconds=predict_time,
            alpha=cfg.alpha,
            **params,
        )


def result_filename(cfg: ExperimentConfig, arrays: dict[str, Any]) -> str:
    if cfg.results_filename:
        return cfg.results_filename
    pca_tag = "no_pca"
    if cfg.use_pca:
        pca_tag = f"pca_x{arrays['X_train_model'].shape[1]}_c{arrays['coef_train_model'].shape[1]}_y{arrays['Y_train_model'].shape[1]}"
        if cfg.task_mode == "parameter_set":
            pca_tag += f"_fw1x{arrays['fw1_X_train_model'].shape[1]}_fw1y{arrays['fw1_Y_train_model'].shape[1]}"
    sweep_tag = f"_{cfg.sweep_id}" if cfg.sweep_id else ""
    pde_tag = cfg.pde_name.replace("/", "_").replace(" ", "_")
    return f"results_{pde_tag}_{cfg.task_mode}_{pca_tag}{sweep_tag}.csv"


def run_experiment(config: ExperimentConfig | dict[str, Any]) -> pd.DataFrame:
    cfg = ExperimentConfig(**config) if isinstance(config, dict) else config
    _, results_dir, _, _ = _paths(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(cfg)
    arrays = preprocess(cfg, prepare_arrays(cfg, data))
    rows: list[dict[str, Any]] = []

    if "vanilla_rbf" in cfg.methods:
        run_vanilla(cfg, arrays, rows, "RBF")
    if "vanilla_matern" in cfg.methods:
        run_vanilla(cfg, arrays, rows, "Matern")
    if cfg.task_mode == "parameter_set":
        if "method_matern" in cfg.methods:
            run_framework1_method(cfg, arrays, rows, "Matern")
        if "method_rbf" in cfg.methods:
            run_framework1_method(cfg, arrays, rows, "RBF")
    else:
        if "method_matern" in cfg.methods:
            run_product_method(cfg, arrays, rows, "Matern x Matern")
        if "method_rbf" in cfg.methods:
            run_product_method(cfg, arrays, rows, "RBF x RBF")

    df = pd.DataFrame(rows)
    for key, value in asdict(cfg).items():
        if key not in df.columns and isinstance(value, (str, int, float, bool, type(None))):
            df[key] = value
    if cfg.save_csv:
        df.to_csv(results_dir / result_filename(cfg, arrays), index=False)
    return df


def with_overrides(config: ExperimentConfig, **overrides) -> ExperimentConfig:
    return replace(config, **overrides)
