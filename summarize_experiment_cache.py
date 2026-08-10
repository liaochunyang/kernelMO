from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CSV result tables from cached experiment predictions."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("experiment_cache"))
    parser.add_argument("--output", type=Path, default=Path("experiment_cache/cache_mean_summary.csv"))
    parser.add_argument(
        "--details-output",
        type=Path,
        default=None,
        help="Optional CSV with one row per cached evaluation item.",
    )
    return parser.parse_args()


def read_cache_file(path: Path) -> tuple[dict, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        errors = np.asarray(data["per_item_relative_error"])
    return metadata, errors


def build_summary(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    detail_rows = []

    for path in sorted(cache_dir.rglob("*.npz")):
        metadata, errors = read_cache_file(path)
        row = {
            "experiment": metadata.get("experiment"),
            "pde_key": metadata.get("pde_key"),
            "pde_name": metadata.get("pde_name"),
            "task_mode": metadata.get("task_mode"),
            "base_dir": metadata.get("base_dir"),
            "split": metadata.get("split_key"),
            "ood_filename": metadata.get("ood_filename"),
            "method": metadata.get("method_key"),
            "kernel": metadata.get("kernel"),
            "pca": metadata.get("use_pca"),
            "n_eval": int(len(errors)),
            "mean_relative_error": float(np.mean(errors)),
            "std_relative_error": float(np.std(errors)),
            "mean_percent": 100 * float(np.mean(errors)),
            "std_percent": 100 * float(np.std(errors)),
            "path": str(path),
        }
        summary_rows.append(row)

        for item_index, error in enumerate(errors):
            detail_rows.append(
                {
                    **{key: row[key] for key in [
                        "experiment",
                        "pde_key",
                        "pde_name",
                        "task_mode",
                        "base_dir",
                        "split",
                        "ood_filename",
                        "method",
                        "kernel",
                        "pca",
                    ]},
                    "item_index": item_index,
                    "relative_error": float(error),
                    "percent_error": 100 * float(error),
                    "path": str(path),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def main() -> None:
    args = parse_args()
    summary, details = build_summary(args.cache_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Wrote summary table with {len(summary)} rows: {args.output}")

    if args.details_output is not None:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        details.to_csv(args.details_output, index=False)
        print(f"Wrote per-item table with {len(details)} rows: {args.details_output}")


if __name__ == "__main__":
    main()
