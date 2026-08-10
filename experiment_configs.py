from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PDE_ORDER = [
    "conservation",
    "diffreacadv",
    "klein_gordon",
    "param_diffreac",
    "param_wave",
]


@dataclass(frozen=True)
class PDEConfig:
    key: str
    label: str
    framework1_name: str
    framework2_name: str
    ov_rbf_gamma: float
    ov_matern_gamma: float
    ov_matern_nu: float
    ov_ps_kw_rbf_gamma: float
    ov_ps_kw_matern_gamma: float
    ov_ps_kw_matern_nu: float
    ov_ps_ku_rbf_gamma: float
    ov_ps_ku_matern_gamma: float
    ov_ps_ku_matern_nu: float
    ov_pca_alpha: bool
    ov_pca_x_dim: int
    ov_pca_y_dim: int
    ps_kernel_o_rbf_gamma: float
    ps_kernel_o_matern_gamma: float
    ps_kernel_o_matern_nu: float
    ps_kw_rbf_gamma: float
    ps_kw_matern_gamma: float
    ps_kw_matern_nu: float
    ps_ku_rbf_gamma: float
    ps_ku_matern_gamma: float
    ps_ku_matern_nu: float
    ps_pca_alpha: bool
    ps_pca_x_dim: int
    ps_pca_coef_dim: int
    ps_pca_y_dim: int


PDE_CONFIGS = {
    "conservation": PDEConfig(
        key="conservation",
        label="Conservation law",
        framework1_name="Conservation_law",
        framework2_name="Conservation_law",
        ov_rbf_gamma=0.1,
        ov_matern_gamma=1.0,
        ov_matern_nu=2.5,
        ov_ps_kw_rbf_gamma=100.0,
        ov_ps_kw_matern_gamma=1.0,
        ov_ps_kw_matern_nu=2.5,
        ov_ps_ku_rbf_gamma=100.0,
        ov_ps_ku_matern_gamma=1.0,
        ov_ps_ku_matern_nu=2.5,
        ov_pca_alpha=False,
        ov_pca_x_dim=4,
        ov_pca_y_dim=10,
        ps_kernel_o_rbf_gamma=100.0,
        ps_kernel_o_matern_gamma=1.0,
        ps_kernel_o_matern_nu=2.5,
        ps_kw_rbf_gamma=1.0,
        ps_kw_matern_gamma=1.0,
        ps_kw_matern_nu=2.5,
        ps_ku_rbf_gamma=100.0,
        ps_ku_matern_gamma=100.0,
        ps_ku_matern_nu=1.5,
        ps_pca_alpha=False,
        ps_pca_x_dim=10,
        ps_pca_coef_dim=4,
        ps_pca_y_dim=10,
    ),
    "diffreacadv": PDEConfig(
        key="diffreacadv",
        label="Diffusion-reaction-advection",
        framework1_name="DiffReacAdv",
        framework2_name="DiffReacAdv",
        ov_rbf_gamma=1.0,
        ov_matern_gamma=10.0,
        ov_matern_nu=3.5,
        ov_ps_kw_rbf_gamma=100.0,
        ov_ps_kw_matern_gamma=10.0,
        ov_ps_kw_matern_nu=2.5,
        ov_ps_ku_rbf_gamma=100.0,
        ov_ps_ku_matern_gamma=10.0,
        ov_ps_ku_matern_nu=2.5,
        ov_pca_alpha=False,
        ov_pca_x_dim=5,
        ov_pca_y_dim=10,
        ps_kernel_o_rbf_gamma=10.0,
        ps_kernel_o_matern_gamma=1.0,
        ps_kernel_o_matern_nu=2.5,
        ps_kw_rbf_gamma=1.0,
        ps_kw_matern_gamma=1.0,
        ps_kw_matern_nu=2.5,
        ps_ku_rbf_gamma=100.0,
        ps_ku_matern_gamma=100.0,
        ps_ku_matern_nu=1.5,
        ps_pca_alpha=False,
        ps_pca_x_dim=10,
        ps_pca_coef_dim=5,
        ps_pca_y_dim=10,
    ),
    "klein_gordon": PDEConfig(
        key="klein_gordon",
        label="Nonlinear Klein-Gordon",
        framework1_name="Nonlinear_Klein_Gordon",
        framework2_name="Nonlinear_Klein_Gordon",
        ov_rbf_gamma=1.0,
        ov_matern_gamma=1.0,
        ov_matern_nu=2.5,
        ov_ps_kw_rbf_gamma=100.0,
        ov_ps_kw_matern_gamma=1.0,
        ov_ps_kw_matern_nu=2.5,
        ov_ps_ku_rbf_gamma=100.0,
        ov_ps_ku_matern_gamma=1.0,
        ov_ps_ku_matern_nu=3.5,
        ov_pca_alpha=False,
        ov_pca_x_dim=3,
        ov_pca_y_dim=10,
        ps_kernel_o_rbf_gamma=10.0,
        ps_kernel_o_matern_gamma=1.0,
        ps_kernel_o_matern_nu=2.5,
        ps_kw_rbf_gamma=1.0,
        ps_kw_matern_gamma=1.0,
        ps_kw_matern_nu=2.5,
        ps_ku_rbf_gamma=10.0,
        ps_ku_matern_gamma=100.0,
        ps_ku_matern_nu=2.5,
        ps_pca_alpha=False,
        ps_pca_x_dim=10,
        ps_pca_coef_dim=3,
        ps_pca_y_dim=10,
    ),
    "param_diffreac": PDEConfig(
        key="param_diffreac",
        label="Parametric diffusion-reaction",
        framework1_name="Param_DiffReac",
        framework2_name="Param_DiffReac",
        ov_rbf_gamma=0.1,
        ov_matern_gamma=0.1,
        ov_matern_nu=2.5,
        ov_ps_kw_rbf_gamma=100.0,
        ov_ps_kw_matern_gamma=1.0,
        ov_ps_kw_matern_nu=2.5,
        ov_ps_ku_rbf_gamma=100.0,
        ov_ps_ku_matern_gamma=0.1,
        ov_ps_ku_matern_nu=2.5,
        ov_pca_alpha=True,
        ov_pca_x_dim=10,
        ov_pca_y_dim=10,
        ps_kernel_o_rbf_gamma=100.0,
        ps_kernel_o_matern_gamma=100.0,
        ps_kernel_o_matern_nu=1.5,
        ps_kw_rbf_gamma=1.0,
        ps_kw_matern_gamma=1.0,
        ps_kw_matern_nu=2.5,
        ps_ku_rbf_gamma=1000.0,
        ps_ku_matern_gamma=1000.0,
        ps_ku_matern_nu=2.5,
        ps_pca_alpha=True,
        ps_pca_x_dim=10,
        ps_pca_coef_dim=10,
        ps_pca_y_dim=10,
    ),
    "param_wave": PDEConfig(
        key="param_wave",
        label="Parametric wave",
        framework1_name="Parametric_Wave",
        framework2_name="Param_wave",
        ov_rbf_gamma=1.0,
        ov_matern_gamma=1.0,
        ov_matern_nu=2.5,
        ov_ps_kw_rbf_gamma=1.0,
        ov_ps_kw_matern_gamma=1.0,
        ov_ps_kw_matern_nu=2.5,
        ov_ps_ku_rbf_gamma=1000.0,
        ov_ps_ku_matern_gamma=1.0,
        ov_ps_ku_matern_nu=2.5,
        ov_pca_alpha=True,
        ov_pca_x_dim=10,
        ov_pca_y_dim=10,
        ps_kernel_o_rbf_gamma=10.0,
        ps_kernel_o_matern_gamma=100.0,
        ps_kernel_o_matern_nu=1.5,
        ps_kw_rbf_gamma=1.0,
        ps_kw_matern_gamma=1.0,
        ps_kw_matern_nu=2.5,
        ps_ku_rbf_gamma=1000.0,
        ps_ku_matern_gamma=1000.0,
        ps_ku_matern_nu=2.5,
        ps_pca_alpha=True,
        ps_pca_x_dim=10,
        ps_pca_coef_dim=10,
        ps_pca_y_dim=10,
    ),
}


EXPERIMENT_LABELS = {
    "framework1_operator": "Framework 1: operator-valued",
    "framework2_product": "Framework 2: product-space",
    "framework2_on_framework1_product": "Framework 2 on Framework 1 data",
}


def slug(text: object) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in str(text))
    return "_".join(part for part in cleaned.split("_") if part)


def cache_path(
    cache_dir: Path,
    experiment: str,
    pde_key: str,
    split_key: str,
    method_key: str,
    pca: bool,
) -> Path:
    pca_tag = "pca" if pca else "no_pca"
    return cache_dir / experiment / pde_key / split_key / pca_tag / f"{method_key}.npz"
