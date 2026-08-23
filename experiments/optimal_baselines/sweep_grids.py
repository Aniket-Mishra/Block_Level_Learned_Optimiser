import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_runner_1_1_config import DEFAULT_OUTPUT_DIR as _MAIN_OUTPUT_DIR

OPTIMAL_BASELINES_OUTPUT_DIR = os.path.join(
    os.path.dirname(_MAIN_OUTPUT_DIR) or ".", "optimal_baselines_full_network"
)

FULL_NETWORK = None

_CONV_METHODS = ["ewc_vanilla", "der_pp", "er_ace", "wsn", "lamaml"]
_VIT_ONLY_METHODS = ["l2p", "dytox"]

_VIT_DATASETS = {"cifar100_vit", "tinyimagenet_vit"}

ALL_DATASETS = [
    "splitmnist",
    "rotatedmnist",
    "cifar100",
    "cifar100_resnet",
    "cifar100_vit",
    "tinyimagenet_resnet",
    "tinyimagenet_vit",
]


def methods_for(dataset):
    methods = list(_CONV_METHODS)
    if dataset in _VIT_DATASETS:
        methods += _VIT_ONLY_METHODS
    return methods


def _is_mnist(dataset):
    return "mnist" in dataset


# Published Adam learning rate per method. lamaml builds its own inner-SGD plus meta-Adam
LEARNING_RATE_BY_METHOD = {
    "ewc_vanilla": 1e-3,  # reg/replay map to ~1e-3 under Adam
    "der_pp": 1e-3,
    "er_ace": 1e-3,
    "wsn": 1e-3,
    "l2p": 3e-2,  # Wang 2022: only prompts and head train
    "dytox": 5e-4,  # Douillard 2022, Table 7
}


def _ewc_lambda_axis(dataset):
    return [10.0, 100.0, 1000.0, 10000.0]


def grid_points(dataset, method):
    if method == "lamaml":
        return [{}]  # inherits inner_lr_init / eta / mem from baseline_config

    learning_rate = LEARNING_RATE_BY_METHOD[method]

    if method == "ewc_vanilla":
        return [
            {"baseline_lr": learning_rate, "ewc_lambda": value}
            for value in _ewc_lambda_axis(dataset)
        ]

    return [{"baseline_lr": learning_rate}]


def grid_id(point):
    """Short, stable identifier for a grid point. Used in the per-run file
    name (resume key) and as a column in the results, so a row is traceable
    to the exact override that produced it."""
    parts = []
    for key in sorted(point):
        value = point[key]
        if isinstance(value, float):
            value = f"{value:g}"
        parts.append(f"{key}={value}")
    return ";".join(parts) if parts else "default"


def count_runs(datasets=None, seeds=(0,)):
    datasets = datasets or ALL_DATASETS
    total = 0
    for dataset in datasets:
        for method in methods_for(dataset):
            total += len(grid_points(dataset, method)) * len(seeds)
    return total


if __name__ == "__main__":
    print(f"Full sweep, 1 tuning seed: {count_runs()} runs total")
    for dataset in ALL_DATASETS:
        per_dataset = sum(
            len(grid_points(dataset, method))
            for method in methods_for(dataset)
        )
        print(f"  {dataset:<22} {per_dataset:>3} runs/seed")
