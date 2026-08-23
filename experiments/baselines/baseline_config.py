"""Baseline experiment config

Baselines inherit the same shared config as proposed
BASE_CONFIG + the dataset's feature flags + dataset config + the dataset's
step budget from STEPS_BY_DATASET and BASELINE_DEFAULTS overrides paper-specific keys per method

training_layers:
    None or 'all': baseline trains every parameter.
    list of str : baseline trains only those modules; all other trainable parameters are trained by a plain Adam optimizer running in parallel.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_runner_1_1_config import (
    BASE_CONFIG,
    DATASET_CONFIGS,
    feature_flags_for,
    normalize_training_layers,
    steps_for,
)
from baselines.baseline_handler import compute_equivalent_epochs


SHARED_BASELINE_SETTINGS = {
    "train_mb_size": 32,
    "eval_mb_size": 128,
}

BASELINE_DEFAULTS = {
    "ewc_online": {
        "ewc_lambda": 400.0,
        "ewc_gamma": 1.0,
    },
    "ewc_vanilla": {
        "ewc_lambda": 400.0,
    },
    "der_pp": {
        # CIFAR default (Buzzega 2020); MNIST/Tiny via dataset overrides.
        "der_alpha": 0.1,
        "der_beta": 0.5,
        "mem_size": 500,
    },
    "er_ace": {
        "mem_size": 500,
        "class_il": True,
    },
    "l2p": {
        "l2p_pool_size": 10,
        "l2p_prompt_length": 5,
        "l2p_top_k": 5,
        "l2p_pull_weight": 0.5,  # Wang 2022 (was 0.1, too low)
    },
    "dytox": {
        "dytox_tab_heads": 6,
        "dytox_class_il": False,
    },
    "wsn": {
        # CIFAR default; MNIST/Tiny via dataset overrides. c = fraction kept.
        "sparsity": 0.5,
    },
    "lamaml": {
        # inner_lr_init: CIFAR/Tiny default; MNIST 0.15 via dataset overrides.
        "inner_lr_init": 0.1,
        "alpha_lr_lr": 0.3,  # LaMAML paper (was 1e-3, far too low)
        "mem_size": 200,  # LaMAML paper (was 500)
        "grad_clip_norm": 2.0,
        "second_order": True,  # headline result (was False = first-order)
        "glances": 1,
    },
    # L2L is the base-paper meta-optimizer. Hyperparameters follow the base
    # paper's Appendix A.1, which states one uniform setting for all datasets:
    # inner-loop SGD lr 1e-3, outer Adam lr 1e-4 with weight decay 1e-5, and a
    # tanh clamp in [-3, 3] on the transformer output. Section 5.2: top 60%
    # of importance scores retained (top_K 0.6). Step budgets (steps,
    # warmup_steps, test_steps) still come from the shared per-dataset budget.
    # Note the released base-paper code uses different SplitMNIST defaults
    # (lr 1e-3, lr_inner 1e-2, lambda_l2 1e-4, clamp 1.0); the paper text was
    # chosen deliberately over the code defaults.
    "l2l": {
        "lr": 1e-4,
        "lr_inner": 1e-3,
        "lambda_l2": 1e-5,
        "clamping_transformer": 3.0,
        "top_K": 0.6,
    },
}

VIT_ONLY_METHODS = {"l2p", "dytox"}


BASELINE_DATASET_OVERRIDES = {
    "ewc_vanilla": {
        # Fisher here is the sample-mean of squared grads summed over ALL
        # params (unnormalized), so large backbones need large lambda.
        "cifar100": {"ewc_lambda": 10000.0},
        "cifar100_resnet": {"ewc_lambda": 10000.0},
        "cifar100_vit": {"ewc_lambda": 10000.0},
        "tinyimagenet_resnet": {"ewc_lambda": 10000.0},
        "tinyimagenet_vit": {"ewc_lambda": 10000.0},
    },
    "der_pp": {
        "splitmnist": {"der_alpha": 1.0},
        "rotatedmnist": {"der_alpha": 1.0},
        "tinyimagenet_resnet": {"der_alpha": 0.2},
        "tinyimagenet_vit": {"der_alpha": 0.2},
    },
    "wsn": {
        "splitmnist": {"sparsity": 0.3},
        "rotatedmnist": {"sparsity": 0.3},
        "tinyimagenet_resnet": {"sparsity": 0.1},
        "tinyimagenet_vit": {"sparsity": 0.1},
    },
    "lamaml": {
        "splitmnist": {"inner_lr_init": 0.15},
        "rotatedmnist": {"inner_lr_init": 0.15},
    },
}


def _estimate_task_dataset_size(dataset_name, config):
    base_name = config.get("base_dataset", dataset_name)
    if "mnist" in base_name:
        total_train = 60000
    elif base_name.startswith("cifar100"):
        total_train = 50000
    elif base_name.startswith("tinyimagenet"):
        total_train = 100000
    else:
        total_train = 50000

    num_labels = config.get("num_labels", 10)
    classes_per_task = config.get("n", 2)
    class_groups = max(num_labels // classes_per_task, 1)
    return total_train // class_groups


def _validate_budget(merged, task_dataset_size, base_epochs, warmup_epochs):
    """budget check on steps to epoch, this was with chatgpt"""
    dataset = merged["dataset"]
    budget = steps_for(dataset)
    for key in ("steps", "warmup_steps", "test_steps"):
        if merged.get(key) != budget[key]:
            raise AssertionError(
                f"[budget] {dataset}/{merged['method']}: {key}="
                f"{merged.get(key)} does not match steps_for('{dataset}')"
                f"[{key}]={budget[key]}. Baselines must inherit the shared "
                f"per-dataset budget."
            )
    batch_size = merged.get("train_mb_size", merged.get("batch_size", 32))
    batches_per_epoch = max(1, math.ceil(task_dataset_size / batch_size))
    expected_base = max(1, round(budget["steps"] / batches_per_epoch))
    expected_warmup = max(1, round(budget["warmup_steps"] / batches_per_epoch))
    if (base_epochs, warmup_epochs) != (expected_base, expected_warmup):
        raise AssertionError(
            f"[budget] {dataset}/{merged['method']}: epochs "
            f"({base_epochs}, {warmup_epochs}) != expected "
            f"({expected_base}, {expected_warmup}) for {budget['steps']}/"
            f"{budget['warmup_steps']} steps, task size {task_dataset_size}, "
            f"batch {batch_size}."
        )


def make_baseline_config(dataset, method, training_layers=None):
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'.")

    dataset_cfg = DATASET_CONFIGS[dataset]
    if method in VIT_ONLY_METHODS and dataset_cfg["model_class"] != "ViTSmall":
        return None

    method_defaults = BASELINE_DEFAULTS.get(method, {})
    dataset_overrides = BASELINE_DATASET_OVERRIDES.get(method, {}).get(
        dataset, {}
    )
    normalized_layers = normalize_training_layers(training_layers)

    merged = {
        **BASE_CONFIG,
        **feature_flags_for(dataset),
        **dataset_cfg,
        **steps_for(dataset),
        **SHARED_BASELINE_SETTINGS,
        **method_defaults,
        **dataset_overrides,
        "dataset": dataset,
        "method": method,
        "training_layers": normalized_layers,
    }

    task_dataset_size = _estimate_task_dataset_size(dataset, merged)
    base_epochs, warmup_epochs = compute_equivalent_epochs(
        merged, task_dataset_size
    )
    _validate_budget(merged, task_dataset_size, base_epochs, warmup_epochs)
    merged["train_epochs"] = base_epochs
    merged["warmup_epochs"] = warmup_epochs

    return SimpleNamespace(**merged)


def budget_report(datasets=None):
    """One row per dataset: the shared step budget and what every baseline
    converts it to (epochs). Validates each row by building a real config."""
    rows = []
    for dataset in datasets or sorted(DATASET_CONFIGS):
        job = make_baseline_config(
            dataset, "ewc_vanilla"
        )  # any epoch-based method
        merged = vars(job)
        task_size = _estimate_task_dataset_size(dataset, merged)
        budget = steps_for(dataset)
        rows.append(
            {
                "dataset": dataset,
                "steps": budget["steps"],
                "warmup_steps": budget["warmup_steps"],
                "test_steps": budget["test_steps"],
                "task_dataset_size": task_size,
                "train_mb_size": merged["train_mb_size"],
                "train_epochs": merged["train_epochs"],
                "warmup_epochs": merged["warmup_epochs"],
            }
        )
    return rows


if __name__ == "__main__":
    header = (
        f"{'dataset':<22}{'steps':>8}{'warmup':>8}{'test':>6}"
        f"{'task_size':>11}{'mb':>4}{'epochs':>8}{'warm_ep':>9}"
    )
    print(header)
    for row in budget_report():
        print(
            f"{row['dataset']:<22}{row['steps']:>8}{row['warmup_steps']:>8}"
            f"{row['test_steps']:>6}{row['task_dataset_size']:>11}"
            f"{row['train_mb_size']:>4}{row['train_epochs']:>8}{row['warmup_epochs']:>9}"
        )
    print(
        "\nAll budgets validated (steps inherited per dataset; conversion checked)."
    )
