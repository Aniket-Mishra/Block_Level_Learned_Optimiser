"""Run full-network optimal-baseline hyperparameter sweep.

Examples:
    python -m optimal_baselines.run_sweep --dry-run
    python -m optimal_baselines.run_sweep --seeds 0
    python -m optimal_baselines.run_sweep --datasets cifar100_resnet --methods er_ace wsn --seeds 0
    python -m optimal_baselines.run_sweep --combine-only
"""

import argparse
import concurrent.futures
import glob
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from baselines.baseline_config import make_baseline_config
from run_plans import run_single_job
from optimal_baselines.sweep_grids import (
    ALL_DATASETS,
    FULL_NETWORK,
    OPTIMAL_BASELINES_OUTPUT_DIR,
    grid_id,
    grid_points,
    methods_for,
)


def runs_dir_of(output_dir):
    return os.path.join(output_dir, "runs")


def combined_dir_of(output_dir):
    return os.path.join(output_dir, "combined")


def _run_filename(dataset, method, point_id, seed, steps):
    safe_point = point_id.replace(";", "__").replace("=", "-")
    return f"run_{dataset}_{method}_seed{seed}_steps{steps}_{safe_point}.csv"


def build_specs(datasets, method_filter, seeds):
    """One spec per (dataset, method, grid point, seed). A spec carries the
    base config from make_baseline_config plus the grid point to overlay."""
    specs = []
    for dataset in datasets:
        for method in methods_for(dataset):
            if method_filter and method not in method_filter:
                continue
            base_config = make_baseline_config(dataset, method, FULL_NETWORK)
            if base_config is None:
                continue  # method/model mismatch, e.g. a ViT-only method
            for point in grid_points(dataset, method):
                for seed in seeds:
                    specs.append((dataset, method, point, seed, base_config))
    return specs


def _job_from_spec(dataset, method, point, seed, base_config, output_dir):
    config = dict(vars(base_config))
    config.update(point)
    config["seed"] = seed
    config["training_layers"] = FULL_NETWORK
    config["output_dir"] = output_dir
    config["training_layers_name"] = "full_network"
    config["optimal_baseline_grid_id"] = grid_id(point)
    return SimpleNamespace(**config)


def _result_is_success(result):
    return isinstance(result, dict) and "average_accuracy_final" in result


def _row_from_result(dataset, method, point, seed, result):
    row = {
        "dataset": dataset,
        "method": method,
        "scope": "full_network",
        "grid_id": grid_id(point),
        "seed": seed,
    }
    row.update({f"hp_{key}": value for key, value in point.items()})
    row.update(result)
    return row


def _execute_one(task):
    """Runs a single pending job and writes its row on success. Returns a
    short status line. Defined at module level so a process pool can pickle
    it. Each run_single_job initializes CUDA in its own process, the same
    pattern the main runner uses, so several share one GPU safely."""
    dataset, method, point, seed, base_config, output_dir, path = task
    job = _job_from_spec(dataset, method, point, seed, base_config, output_dir)
    tag = f"{method} {dataset} seed{seed} steps{base_config.steps} {grid_id(point)}"
    result = run_single_job(job)
    if not _result_is_success(result):
        return f"FAILED {tag} (not recorded, will retry next launch)"
    row = _row_from_result(dataset, method, point, seed, result)
    pd.DataFrame([row]).to_csv(path, index=False)
    return f"done {tag} -> avg_acc={result.get('average_accuracy_final')}"


def _pending_tasks(specs, output_dir):
    """Specs that have no per-run file yet, each packed with its output path."""
    runs_dir = runs_dir_of(output_dir)
    os.makedirs(runs_dir, exist_ok=True)
    tasks = []
    for dataset, method, point, seed, base_config in specs:
        filename = _run_filename(
            dataset, method, grid_id(point), seed, base_config.steps
        )
        path = os.path.join(runs_dir, filename)
        if os.path.exists(path):
            continue
        tasks.append((dataset, method, point, seed, base_config, output_dir, path))
    return tasks


def run_pending(specs, output_dir, max_workers):
    """Runs every spec without an existing per-run file, writing one CSV row
    per success as it finishes so progress survives an interruption. With
    max_workers > 1 the runs share the GPU through a process pool."""
    tasks = _pending_tasks(specs, output_dir)
    print(f"{len(specs)} runs total, {len(specs) - len(tasks)} already done, "
          f"{len(tasks)} to run, max_workers={max_workers}.")
    print(f"writing per-run CSVs to {runs_dir_of(output_dir)}")

    if not tasks:
        return

    if max_workers == 1:
        for index, task in enumerate(tasks):
            print(f"[{index + 1}/{len(tasks)}] {_execute_one(task)}")
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_execute_one, task): task for task in tasks}
        for index, future in enumerate(concurrent.futures.as_completed(futures)):
            print(f"[{index + 1}/{len(tasks)}] {future.result()}")


def combine_runs(output_dir):
    """Concatenates every per-run CSV into one timestamped combined CSV."""
    runs_dir = runs_dir_of(output_dir)
    run_files = sorted(glob.glob(os.path.join(runs_dir, "run_*.csv")))
    if not run_files:
        print(f"No per-run files in {runs_dir}; nothing to combine.")
        return None

    frame = pd.concat(
        [pd.read_csv(path) for path in run_files], ignore_index=True
    )
    combined_dir = combined_dir_of(output_dir)
    os.makedirs(combined_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    combined_path = os.path.join(
        combined_dir, f"optimal_baselines_{timestamp}.csv"
    )
    frame.to_csv(combined_path, index=False)
    print(f"Combined {len(run_files)} runs into {combined_path}")
    return combined_path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--output-dir", default=OPTIMAL_BASELINES_OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Parallel runs sharing the GPU. 1 is sequential.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the runs and exit without training.")
    parser.add_argument("--combine-only", action="store_true",
                        help="Rebuild the combined CSV from existing per-run files.")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    output_dir = arguments.output_dir
    print(f"optimal baselines output dir: {output_dir}")

    if arguments.combine_only:
        combine_runs(output_dir)
        return

    datasets = arguments.datasets or ALL_DATASETS
    specs = build_specs(datasets, arguments.methods, arguments.seeds)

    if arguments.dry_run:
        print(f"{len(specs)} runs planned:")
        for dataset, method, point, seed, base_config in specs:
            print(f"  {method:<12} {dataset:<22} seed={seed} "
                  f"steps={base_config.steps} epochs={base_config.train_epochs} "
                  f"{grid_id(point)}")
        return

    run_pending(specs, output_dir, arguments.max_workers)
    combine_runs(output_dir)


if __name__ == "__main__":
    main()
