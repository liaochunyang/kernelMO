from __future__ import annotations

import argparse
import itertools
import random
from dataclasses import replace
from pathlib import Path


def log_grid():
    return [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def default_grid(task_mode: str):
    values = log_grid()
    if task_mode == "parameter_set":
        for pde, fw1_rbf, fw1_matern in itertools.product(
            ["Conservation_law", "DiffReacAdv", "Nonlinear_Klein_Gordon", "Param_DiffReac", "Parametric_Wave"],
            values,
            values,
        ):
            yield {
                "pde_name": pde,
                "framework1_rbf_length_scale": fw1_rbf,
                "framework1_matern_length_scale": fw1_matern,
                "methods": ("method_rbf", "method_matern"),
            }
    else:
        for pde, coef_scale, state_scale in itertools.product(
            ["Conservation_law", "DiffReacAdv", "Nonlinear_Klein_Gordon", "Param_DiffReac", "Param_wave"],
            values,
            values,
        ):
            yield {
                "pde_name": pde,
                "product_rbf_coef_length_scale": coef_scale,
                "product_rbf_state_length_scale": state_scale,
                "product_matern_coef_length_scale": coef_scale,
                "product_matern_state_length_scale": state_scale,
                "methods": ("method_rbf", "method_matern"),
            }


def random_grid(task_mode: str, n_trials: int, seed: int):
    rng = random.Random(seed)
    values = log_grid()
    choices = list(default_grid(task_mode))
    for _ in range(n_trials):
        item = dict(rng.choice(choices))
        if task_mode == "sample":
            item["product_matern_coef_nu"] = rng.choice([0.5, 1.5, 2.5])
            item["product_matern_state_nu"] = rng.choice([0.5, 1.5, 2.5])
        else:
            item["framework1_matern_nu"] = rng.choice([0.5, 1.5, 2.5])
        yield item


def run_grid(args):
    from experiment_runner import ExperimentConfig

    base = ExperimentConfig(
        task_mode=args.task_mode,
        use_pca=args.use_pca,
        use_ood=not args.no_ood,
        validation_fraction=args.validation_fraction,
        save_csv=False,
    )
    candidates = list(default_grid(args.task_mode))
    if args.max_runs is not None:
        candidates = candidates[: args.max_runs]
    return run_candidates(base, candidates, args.output)


def run_random(args):
    from experiment_runner import ExperimentConfig

    base = ExperimentConfig(
        task_mode=args.task_mode,
        use_pca=args.use_pca,
        use_ood=not args.no_ood,
        validation_fraction=args.validation_fraction,
        save_csv=False,
    )
    return run_candidates(base, list(random_grid(args.task_mode, args.n_trials, args.seed)), args.output)


def run_candidates(base: ExperimentConfig, candidates: list[dict], output: str):
    import pandas as pd

    from experiment_runner import run_experiment

    frames = []
    for index, overrides in enumerate(candidates):
        cfg = replace(base, sweep_id=f"sweep_{index:04d}", **overrides)
        print(f"[{index + 1}/{len(candidates)}] {cfg.pde_name} {cfg.sweep_id}")
        df = run_experiment(cfg)
        frames.append(df)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_optuna(args):
    import pandas as pd

    from experiment_runner import ExperimentConfig, run_experiment

    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Optuna is not installed. Install with: pip install optuna") from exc

    pdes = (
        ["Conservation_law", "DiffReacAdv", "Nonlinear_Klein_Gordon", "Param_DiffReac", "Parametric_Wave"]
        if args.task_mode == "parameter_set"
        else ["Conservation_law", "DiffReacAdv", "Nonlinear_Klein_Gordon", "Param_DiffReac", "Param_wave"]
    )
    base = ExperimentConfig(
        task_mode=args.task_mode,
        pde_name=args.pde or pdes[0],
        use_ood=not args.no_ood,
        validation_fraction=args.validation_fraction,
        save_csv=False,
    )
    frames = []

    def objective(trial):
        if args.task_mode == "parameter_set":
            cfg = replace(
                base,
                sweep_id=f"optuna_{trial.number:04d}",
                framework1_rbf_length_scale=trial.suggest_float("framework1_rbf_length_scale", 1e-2, 1e2, log=True),
                framework1_matern_length_scale=trial.suggest_float("framework1_matern_length_scale", 1e-2, 1e2, log=True),
                framework1_matern_nu=trial.suggest_categorical("framework1_matern_nu", [0.5, 1.5, 2.5]),
                methods=("method_rbf", "method_matern"),
            )
        else:
            cfg = replace(
                base,
                sweep_id=f"optuna_{trial.number:04d}",
                product_rbf_coef_length_scale=trial.suggest_float("product_rbf_coef_length_scale", 1e-2, 1e2, log=True),
                product_rbf_state_length_scale=trial.suggest_float("product_rbf_state_length_scale", 1e-2, 1e2, log=True),
                product_matern_coef_length_scale=trial.suggest_float("product_matern_coef_length_scale", 1e-2, 1e2, log=True),
                product_matern_state_length_scale=trial.suggest_float("product_matern_state_length_scale", 1e-2, 1e2, log=True),
                product_matern_coef_nu=trial.suggest_categorical("product_matern_coef_nu", [0.5, 1.5, 2.5]),
                product_matern_state_nu=trial.suggest_categorical("product_matern_state_nu", [0.5, 1.5, 2.5]),
                methods=("method_rbf", "method_matern"),
            )
        df = run_experiment(cfg)
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_csv(args.output, index=False)
        objective_split = "val" if df["split"].eq("val").any() else "test"
        objective_rows = df[df["split"].eq(objective_split)]
        return float(objective_rows["mean_relative_error"].min())

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


def run_skopt(args):
    import pandas as pd

    from experiment_runner import ExperimentConfig, run_experiment

    try:
        from skopt import gp_minimize
        from skopt.space import Categorical, Real
    except ImportError as exc:
        raise SystemExit("scikit-optimize is not installed. Install with: pip install scikit-optimize") from exc

    base = ExperimentConfig(
        task_mode=args.task_mode,
        pde_name=args.pde or "Conservation_law",
        use_ood=not args.no_ood,
        validation_fraction=args.validation_fraction,
        save_csv=False,
    )
    frames = []

    if args.task_mode == "parameter_set":
        dimensions = [
            Real(1e-2, 1e2, prior="log-uniform", name="framework1_rbf_length_scale"),
            Real(1e-2, 1e2, prior="log-uniform", name="framework1_matern_length_scale"),
            Categorical([0.5, 1.5, 2.5], name="framework1_matern_nu"),
        ]
    else:
        dimensions = [
            Real(1e-2, 1e2, prior="log-uniform", name="product_rbf_coef_length_scale"),
            Real(1e-2, 1e2, prior="log-uniform", name="product_rbf_state_length_scale"),
            Real(1e-2, 1e2, prior="log-uniform", name="product_matern_coef_length_scale"),
            Real(1e-2, 1e2, prior="log-uniform", name="product_matern_state_length_scale"),
            Categorical([0.5, 1.5, 2.5], name="product_matern_coef_nu"),
            Categorical([0.5, 1.5, 2.5], name="product_matern_state_nu"),
        ]

    def objective(values):
        trial_id = len(frames)
        overrides = {dim.name: value for dim, value in zip(dimensions, values)}
        cfg = replace(base, sweep_id=f"skopt_{trial_id:04d}", methods=("method_rbf", "method_matern"), **overrides)
        df = run_experiment(cfg)
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_csv(args.output, index=False)
        objective_split = "val" if df["split"].eq("val").any() else "test"
        return float(df[df["split"].eq(objective_split)]["mean_relative_error"].min())

    result = gp_minimize(objective, dimensions, n_calls=args.n_trials, random_state=args.seed)
    print("Best value:", result.fun)
    print("Best params:", result.x)


def parse_args():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweeps for PDE kernel experiments.")
    parser.add_argument("--mode", choices=["grid", "random", "optuna", "skopt"], default="random")
    parser.add_argument("--task-mode", choices=["sample", "parameter_set"], default="sample")
    parser.add_argument("--pde", default=None, help="PDE for optuna/skopt modes. Grid/random cover all PDEs by default.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--max-runs", type=int, default=None, help="Limit grid runs for quick tests.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-pca", action="store_true")
    parser.add_argument("--no-ood", action="store_true")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help="Fraction of the original test split to reserve for validation. Optimizers use val when present.",
    )
    parser.add_argument("--output", default="sweep_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "grid":
        run_grid(args)
    elif args.mode == "random":
        run_random(args)
    elif args.mode == "optuna":
        run_optuna(args)
    elif args.mode == "skopt":
        run_skopt(args)


if __name__ == "__main__":
    main()
