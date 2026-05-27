from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SampleData:
    x_train: np.ndarray
    coef_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    coef_test: np.ndarray
    y_test: np.ndarray
    x_ood: np.ndarray | None = None
    coef_ood: np.ndarray | None = None
    y_ood: np.ndarray | None = None
    y_train_mean: np.ndarray | None = None
    y_train_std: np.ndarray | None = None
    y_test_mean: np.ndarray | None = None
    y_test_std: np.ndarray | None = None
    y_ood_mean: np.ndarray | None = None
    y_ood_std: np.ndarray | None = None


@dataclass
class Normalizers:
    x_mean: np.ndarray
    x_std: np.ndarray
    coef_mean: np.ndarray
    coef_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def normalize_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def normalize_coef(self, coef: np.ndarray) -> np.ndarray:
        return (coef - self.coef_mean) / self.coef_std

    def normalize_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def denormalize_y_torch(self, y: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.y_mean, dtype=y.dtype, device=y.device)
        std = torch.as_tensor(self.y_std, dtype=y.dtype, device=y.device)
        return y * std + mean


def load_sample_data(
    framework: str = "Framework2",
    pde: str = "Conservation_law",
    data_subdir: str = "dataset_simple",
    data_filename: str = "solutions.h5",
    ood_filename: str = "ood.h5",
    train_size: int | None = 10000,
    train_fraction: float = 0.8,
    input_time_index: int = 0,
    output_start_index: int | None = None,
    output_step: int = 1,
    x_num_model: int | None = None,
    channel_index: int = 0,
    data_key: str = "data",
    coeff_key: str = "coeffs",
    use_ood: bool = True,
) -> SampleData:
    data_path = resolve_data_path(framework, pde, data_subdir, data_filename)
    with h5py.File(data_path, "r") as f:
        dataset = f[data_key][:]
        coef = f[coeff_key][:]

    n_train = train_size if train_size is not None else int(train_fraction * dataset.shape[0])
    x_idx = spatial_indices(dataset.shape[2], x_num_model)
    output_start = input_time_index if output_start_index is None else output_start_index
    time_idx = np.arange(output_start, dataset.shape[1], output_step)
    x = dataset[:, input_time_index, x_idx, channel_index]
    y = dataset[:, time_idx][:, :, x_idx, channel_index]

    sample_data = SampleData(
        x_train=x[:n_train],
        coef_train=coef[:n_train],
        y_train=y[:n_train],
        x_test=x[n_train:],
        coef_test=coef[n_train:],
        y_test=y[n_train:],
    )

    ood_path = resolve_data_path(framework, pde, data_subdir, ood_filename, must_exist=False)
    if use_ood and ood_path.exists():
        with h5py.File(ood_path, "r") as f:
            dataset_ood = f[data_key][:]
            coef_ood = f[coeff_key][:]
        sample_data.x_ood = dataset_ood[:, input_time_index, x_idx, channel_index]
        sample_data.coef_ood = coef_ood
        sample_data.y_ood = dataset_ood[:, time_idx][:, :, x_idx, channel_index]

    return sample_data


def resolve_data_path(
    framework: str,
    pde: str,
    data_subdir: str,
    filename: str,
    must_exist: bool = True,
) -> Path:
    """Locate an h5 file, trying ``<framework>/<pde>/<data_subdir>/<file>`` first
    and falling back to ``<framework>/<pde>/<file>`` (Framework1 stores the data
    directly in the PDE folder, Framework2 under ``dataset_simple``)."""
    base = Path(framework) / pde
    candidates = [base / data_subdir / filename, base / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if must_exist:
        tried = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError(f"Could not find {filename}. Tried: {tried}")
    return candidates[0]


def spatial_indices(n_x: int, x_num_model: int | None) -> np.ndarray:
    if x_num_model is None or x_num_model >= n_x:
        return np.arange(n_x)
    if x_num_model <= 0:
        raise ValueError("x_num_model must be positive.")
    return np.linspace(0, n_x - 1, num=x_num_model).round().astype(int)


def fit_normalizers(data: SampleData, eps: float = 1e-6) -> Normalizers:
    return Normalizers(
        x_mean=data.x_train.mean(axis=0, keepdims=True),
        x_std=data.x_train.std(axis=0, keepdims=True) + eps,
        coef_mean=data.coef_train.mean(axis=0, keepdims=True),
        coef_std=data.coef_train.std(axis=0, keepdims=True) + eps,
        y_mean=data.y_train.mean(axis=0, keepdims=True),
        y_std=data.y_train.std(axis=0, keepdims=True) + eps,
    )


class PDESampleDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        coef: np.ndarray,
        y: np.ndarray,
        y_mean: np.ndarray | None = None,
        y_std: np.ndarray | None = None,
    ):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.coef = torch.as_tensor(coef, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.y_mean = torch.as_tensor(y_mean, dtype=torch.float32) if y_mean is not None else None
        self.y_std = torch.as_tensor(y_std, dtype=torch.float32) if y_std is not None else None

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        item = {
            "x": self.x[index],
            "coef": self.coef[index],
            "y": self.y[index],
        }
        if self.y_mean is not None and self.y_std is not None:
            item["y_mean"] = self.y_mean[index]
            item["y_std"] = self.y_std[index]
        return item


def normalize_data(data: SampleData, normalizers: Normalizers, mode: str = "global", eps: float = 1e-6) -> SampleData:
    if mode == "none":
        return data
    if mode == "per_sample_input":
        return normalize_data_per_sample_input(data, normalizers, eps=eps)
    if mode != "global":
        raise ValueError("normalization mode must be 'global', 'per_sample_input', or 'none'.")
    return SampleData(
        x_train=normalizers.normalize_x(data.x_train),
        coef_train=normalizers.normalize_coef(data.coef_train),
        y_train=normalizers.normalize_y(data.y_train),
        x_test=normalizers.normalize_x(data.x_test),
        coef_test=normalizers.normalize_coef(data.coef_test),
        y_test=normalizers.normalize_y(data.y_test),
        x_ood=normalizers.normalize_x(data.x_ood) if data.x_ood is not None else None,
        coef_ood=normalizers.normalize_coef(data.coef_ood) if data.coef_ood is not None else None,
        y_ood=normalizers.normalize_y(data.y_ood) if data.y_ood is not None else None,
    )


def normalize_data_per_sample_input(data: SampleData, normalizers: Normalizers, eps: float = 1e-6) -> SampleData:
    def normalize_split(x, y):
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + eps
        x_norm = (x - mean) / std
        y_norm = (y - mean[:, None, :]) / std[:, None, :]
        return x_norm, y_norm, mean[:, None, :], std[:, None, :]

    x_train, y_train, train_mean, train_std = normalize_split(data.x_train, data.y_train)
    x_test, y_test, test_mean, test_std = normalize_split(data.x_test, data.y_test)
    if data.x_ood is not None:
        x_ood, y_ood, ood_mean, ood_std = normalize_split(data.x_ood, data.y_ood)
    else:
        x_ood = y_ood = ood_mean = ood_std = None

    return SampleData(
        x_train=x_train,
        coef_train=normalizers.normalize_coef(data.coef_train),
        y_train=y_train,
        x_test=x_test,
        coef_test=normalizers.normalize_coef(data.coef_test),
        y_test=y_test,
        x_ood=x_ood,
        coef_ood=normalizers.normalize_coef(data.coef_ood) if data.coef_ood is not None else None,
        y_ood=y_ood,
        y_train_mean=train_mean,
        y_train_std=train_std,
        y_test_mean=test_mean,
        y_test_std=test_std,
        y_ood_mean=ood_mean,
        y_ood_std=ood_std,
    )
