from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import (
    PDESampleDataset,
    discover_ood_files,
    fit_normalizers,
    load_ood_split,
    load_sample_data,
    normalize_data,
    normalize_ood_split,
)
from models import (
    ConcatDeepONet,
    FNO1d,
    MIONet,
    MNO,
    NeuralOperatorFNO2d,
    TensorizedConcatDeepONet,
    TensorizedDeepONet,
)

# Models that produce point-wise predictions and therefore support trunk
# sub-sampling via `deeponet_loss`.
TENSORIZED_MODELS = {"deeponet", "deeponet_nocoef", "mionet", "mno"}


def relative_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.sqrt(torch.sum((pred - target) ** 2, dim=tuple(range(1, pred.ndim))))
    denominator = torch.sqrt(torch.sum(target**2, dim=tuple(range(1, target.ndim))))
    return numerator / denominator


def make_model(args, n_x: int, coef_dim: int, n_t: int):
    if args.model == "fno":
        return FNO1d(
            n_x=n_x,
            coef_dim=coef_dim,
            n_t=n_t,
            width=args.width,
            modes=args.modes,
            depth=args.depth,
        )
    if args.model == "neuralop_fno":
        return NeuralOperatorFNO2d(
            n_x=n_x,
            coef_dim=coef_dim,
            n_t=n_t,
            hidden_channels=args.width,
            modes_t=args.modes_t,
            modes_x=args.modes,
            n_layers=args.depth,
        )
    if args.model == "deeponet":
        return TensorizedConcatDeepONet(
            n_x=n_x,
            coef_dim=coef_dim,
            n_t=n_t,
            num_trunk=args.num_trunk,
            num_branch=args.num_branch,
            hidden_dim=args.hidden_dim,
            branch_depth=args.branch_depth,
            trunk_depth=args.trunk_depth,
        )
    if args.model == "deeponet_nocoef":
        return TensorizedDeepONet(
            n_x=n_x,
            coef_dim=coef_dim,
            n_t=n_t,
            num_trunk=args.num_trunk,
            num_branch=args.num_branch,
            hidden_dim=args.hidden_dim,
            branch_depth=args.branch_depth,
            trunk_depth=args.trunk_depth,
        )
    if args.model == "mionet":
        return MIONet(
            n_x=n_x,
            coef_dim=coef_dim,
            n_t=n_t,
            latent=args.latent_dim,
            hidden_dim=args.hidden_dim,
            branch_depth=args.branch_depth,
            trunk_depth=args.trunk_depth,
        )
    if args.model == "mno":
        return MNO(
            n_x=n_x,
            coef_dim=coef_dim,
            n_t=n_t,
            num_leaf=args.num_leaf,
            num_trunk=args.num_trunk,
            num_branch=args.num_branch,
            hidden_dim=args.hidden_dim,
            branch_depth=args.branch_depth,
            trunk_depth=args.trunk_depth,
            leaf_depth=args.leaf_depth,
        )
    if args.model == "simple_deeponet":
        return ConcatDeepONet(
            n_x=n_x,
            coef_dim=coef_dim,
            n_t=n_t,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            branch_depth=args.branch_depth,
            trunk_depth=args.trunk_depth,
        )
    raise ValueError(f"Unknown model: {args.model}")


def deeponet_loss(model, batch, args, device):
    x = batch["x"].to(device)
    coef = batch["coef"].to(device)
    y = batch["y"].to(device)
    batch_size, n_t, n_x = y.shape

    if args.trunk_batch_size <= 0 or args.trunk_batch_size >= n_t * n_x:
        pred = model(x, coef)
        return nn.functional.mse_loss(pred, y)

    flat_y = y.reshape(batch_size, -1)
    indices = torch.randint(0, n_t * n_x, (args.trunk_batch_size,), device=device)
    t_idx = indices // n_x
    x_idx = indices % n_x
    coords = torch.stack(
        [
            t_idx.to(torch.float32) / max(n_t - 1, 1),
            x_idx.to(torch.float32) / max(n_x - 1, 1),
        ],
        dim=-1,
    )
    pred = model(x, coef, coords=coords)
    target = flat_y[:, indices]
    return nn.functional.mse_loss(pred, target)


def train_one_epoch(model, loader, optimizer, args, device):
    model.train()
    losses = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        if args.model in TENSORIZED_MODELS:
            loss = deeponet_loss(model, batch, args, device)
        else:
            x = batch["x"].to(device)
            coef = batch["coef"].to(device)
            y = batch["y"].to(device)
            pred = model(x, coef)
            loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def predict_full_grid(model, x, coef, n_t, n_x, point_chunk):
    """Predict the full (t, x) grid, splitting the trunk points into chunks.

    Tensorized models contract over a (batch, points, modes, modes) intermediate,
    so evaluating the whole n_t*n_x grid at once can OOM at large sizes. Running
    the points in chunks of ``point_chunk`` bounds that intermediate the same way
    ``deeponet_loss`` subsamples trunk points during training. ``point_chunk<=0``
    (or a chunk covering the whole grid) falls back to a single full-grid call.
    Returns predictions shaped (batch, n_t, n_x).
    """
    coords = model.coordinate_grid(x.device, x.dtype)  # (n_t*n_x, 2)
    total = coords.shape[0]
    if point_chunk <= 0 or point_chunk >= total:
        return model(x, coef)
    preds = [
        model(x, coef, coords=coords[start:start + point_chunk]).reshape(x.shape[0], -1)
        for start in range(0, total, point_chunk)
    ]
    return torch.cat(preds, dim=1).view(x.shape[0], n_t, n_x)


@torch.no_grad()
def evaluate(model, dataset, normalizers, args, device, split):
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False)
    model.eval()
    # FNO variants emit the full grid in one cheap pass and take no `coords`;
    # the tensorized models do, so chunk their trunk points to bound memory.
    supports_coords = hasattr(model, "coordinate_grid")
    errors = []
    start = perf_counter()
    for batch in loader:
        x = batch["x"].to(device)
        coef = batch["coef"].to(device)
        y_norm = batch["y"].to(device)
        if supports_coords:
            n_t, n_x = y_norm.shape[1], y_norm.shape[2]
            pred_norm = predict_full_grid(model, x, coef, n_t, n_x, args.eval_trunk_batch_size)
        else:
            pred_norm = model(x, coef)
        if "y_mean" in batch and "y_std" in batch:
            y_mean = batch["y_mean"].to(device)
            y_std = batch["y_std"].to(device)
            pred = pred_norm * y_std + y_mean
            target = y_norm * y_std + y_mean
        elif args.normalization == "none":
            pred = pred_norm
            target = y_norm
        else:
            pred = normalizers.denormalize_y_torch(pred_norm)
            target = normalizers.denormalize_y_torch(y_norm)
        errors.append(relative_error(pred, target).detach().cpu())
    elapsed = perf_counter() - start
    errors = torch.cat(errors).numpy()
    return {
        "split": split,
        "mean_relative_error": float(errors.mean()),
        "std_relative_error": float(errors.std()),
        "predict_time_seconds": elapsed,
        "n_eval": int(errors.shape[0]),
    }


def run_single(args) -> pd.DataFrame:
    """Train one model on one (framework, pde) split and return the result rows.

    Does not write any CSV; callers (``main`` or ``run_comparison``) decide how
    to persist the returned DataFrame. A non-positive ``train_size`` means
    "use ``train_fraction`` of the dataset" (the Framework1 convention).
    """
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_size = args.train_size if args.train_size and args.train_size > 0 else None
    raw = load_sample_data(
        framework=args.framework,
        pde=args.pde,
        data_root=args.data_root,
        data_subdir=args.data_subdir,
        train_size=train_size,
        train_fraction=args.train_fraction,
        output_start_index=args.output_start_index,
        output_step=args.output_step,
        x_num_model=args.x_num_model,
        use_ood=False,  # OOD files are loaded separately below to support several per PDE.
    )
    normalizers = fit_normalizers(raw)
    data = normalize_data(raw, normalizers, mode=args.normalization)

    train_dataset = PDESampleDataset(
        data.x_train, data.coef_train, data.y_train, data.y_train_mean, data.y_train_std
    )
    test_dataset = PDESampleDataset(
        data.x_test, data.coef_test, data.y_test, data.y_test_mean, data.y_test_std
    )
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    if args.no_ood:
        ood_files = []
    elif args.ood_filenames == ["auto"]:
        ood_files = discover_ood_files(args.framework, args.pde, args.data_root, args.data_subdir)
    else:
        ood_files = args.ood_filenames

    n_x = data.x_train.shape[1]
    coef_dim = data.coef_train.shape[1]
    n_t = data.y_train.shape[1]
    model = make_model(args, n_x=n_x, coef_dim=coef_dim, n_t=n_t).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_train = perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, args, device)
        history.append({"epoch": epoch, "train_mse": loss})
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:04d} train_mse={loss:.4e}")
    train_time = perf_counter() - start_train

    test_result = evaluate(model, test_dataset, normalizers, args, device, "test")
    test_result["ood_file"] = ""
    rows = [test_result]
    for ood_filename in ood_files:
        split = load_ood_split(
            framework=args.framework,
            pde=args.pde,
            ood_filename=ood_filename,
            data_root=args.data_root,
            data_subdir=args.data_subdir,
            output_start_index=args.output_start_index,
            output_step=args.output_step,
            x_num_model=args.x_num_model,
        )
        if split is None:
            continue
        x_ood, coef_ood, y_ood = split
        x_ood, coef_ood, y_ood, y_mean, y_std = normalize_ood_split(
            x_ood, coef_ood, y_ood, normalizers, mode=args.normalization
        )
        ood_dataset = PDESampleDataset(x_ood, coef_ood, y_ood, y_mean, y_std)
        result = evaluate(model, ood_dataset, normalizers, args, device, "ood")
        result["ood_file"] = ood_filename
        rows.append(result)

    result_df = pd.DataFrame(rows)
    result_df.insert(0, "model", args.model)
    result_df.insert(0, "pde", args.pde)
    result_df.insert(0, "framework", args.framework)
    result_df["n_train"] = len(train_dataset)
    result_df["n_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result_df["train_time_seconds"] = train_time
    result_df["epochs"] = args.epochs
    result_df["lr"] = args.lr
    result_df["batch_size"] = args.batch_size
    result_df["num_trunk"] = args.num_trunk
    result_df["num_branch"] = args.num_branch
    result_df["num_leaf"] = args.num_leaf if args.model == "mno" else np.nan

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "history": history,
            },
            checkpoint,
        )
        print(f"Saved checkpoint to {checkpoint}")

    if args.history_output:
        Path(args.history_output).write_text(json.dumps(history, indent=2))

    return result_df


def main():
    args = parse_args()
    result_df = run_single(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output, index=False)
    print(result_df)
    print(f"Saved results to {output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train FNO, DeepONet, MIONet, or MNO on PDE sample data.")
    parser.add_argument(
        "--model",
        choices=["fno", "neuralop_fno", "deeponet", "deeponet_nocoef", "mionet", "mno", "simple_deeponet"],
        required=True,
    )
    parser.add_argument("--framework", default="Framework2")
    parser.add_argument("--pde", default="Conservation_law")
    parser.add_argument("--data-root", default=".", help="Root holding <framework>/<pde>/... (e.g. /home/shared/dataset/KernelMOL).")
    parser.add_argument("--data-subdir", default="dataset_simple", help="Subfolder holding solutions.h5/ood.h5 (falls back to the PDE folder root).")
    parser.add_argument("--train-size", type=int, default=10000, help="First-N training samples. Use <=0 to fall back to --train-fraction.")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--x-num-model", type=int, default=None, help="Optional number of spatial points to keep.")
    parser.add_argument("--output-start-index", type=int, default=None, help="First output time index. Defaults to input time index.")
    parser.add_argument("--output-step", type=int, default=1, help="Keep every k-th output time step.")
    parser.add_argument("--normalization", choices=["global", "per_sample_input", "none"], default="global")
    parser.add_argument("--no-ood", action="store_true")
    parser.add_argument("--ood-filenames", nargs="*", default=["auto"],
                        help="OOD h5 basenames to evaluate. 'auto' (default) discovers every ood*.h5 for the PDE.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--modes-t", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-trunk", type=int, default=100)
    parser.add_argument("--num-branch", type=int, default=100)
    parser.add_argument("--num-leaf", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--branch-depth", type=int, default=3)
    parser.add_argument("--trunk-depth", type=int, default=3)
    parser.add_argument("--leaf-depth", type=int, default=2)
    parser.add_argument("--trunk-batch-size", type=int, default=2048)
    parser.add_argument("--eval-trunk-batch-size", type=int, default=1024,
                        help="Trunk points per chunk during full-grid evaluation "
                             "(<=0 evaluates the whole grid at once). Bounds the "
                             "tensorized-model einsum memory; tensorized models only.")
    parser.add_argument("--output", default="neural_network_experiments/results.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--history-output", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
