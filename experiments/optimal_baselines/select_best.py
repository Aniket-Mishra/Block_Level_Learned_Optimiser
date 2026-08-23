"""Rank the swept configs and write a summary, for analysis.
    python -m optimal_baselines.select_best
    python -m optimal_baselines.select_best --seeds 0 42 67
"""

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from optimal_baselines.sweep_grids import OPTIMAL_BASELINES_OUTPUT_DIR

ACCURACY = "average_accuracy_final"
BACKWARD = "bwt_avg_over_time"
FORWARD = "mean_future_accuracy_over_time"
GROUP_KEYS = ["dataset", "method", "grid_id"]


def load_runs(output_dir):
    run_files = glob.glob(os.path.join(output_dir, "runs", "run_*.csv"))
    if not run_files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in run_files], ignore_index=True)


def average_over_seeds(runs, seeds):
    selected = runs[runs["seed"].isin(seeds)].copy()
    metrics = [ACCURACY, BACKWARD, FORWARD]
    aggregated = (
        selected.groupby(GROUP_KEYS, dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            **{metric: (metric, "mean") for metric in metrics},
        )
        .reset_index()
    )
    return aggregated


def best_per_method(aggregated):
    ranked = aggregated.sort_values(
        [ACCURACY, BACKWARD, FORWARD], ascending=False
    )
    return ranked.groupby(["dataset", "method"], dropna=False).head(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OPTIMAL_BASELINES_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    arguments = parser.parse_args()

    runs = load_runs(arguments.output_dir)
    if runs.empty:
        print(f"No per-run files under {arguments.output_dir}/runs. "
              "Run run_sweep.py first.")
        return

    aggregated = average_over_seeds(runs, arguments.seeds)
    if aggregated.empty:
        print(f"No runs for seeds {arguments.seeds}.")
        return

    winners = best_per_method(aggregated)

    print(f"\nBest grid point per (dataset, method), averaged over "
          f"seed(s) {arguments.seeds}:\n")
    header = f"{'dataset':<22}{'method':<12}{'acc':>8}{'bwt':>8}{'fwt':>8}  grid_id"
    print(header)
    print("-" * len(header))
    for _, row in winners.sort_values(["dataset", "method"]).iterrows():
        print(f"{row['dataset']:<22}{row['method']:<12}"
              f"{row[ACCURACY]:>8.4f}{row[BACKWARD]:>8.4f}"
              f"{row[FORWARD]:>8.4f}  {row['grid_id']}")

    summary_path = os.path.join(arguments.output_dir, "best_per_method.csv")
    winners.sort_values(["dataset", "method"]).to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
