import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

import dataset
import utils
from models.base_models import resolve_training_layer_tokens


def base_dataset_name(config):
    return config.get("base_dataset") or config["dataset"]


def load_split_datasets(config):
    """Returns (train_datasets, test_datasets, task_labels, n_tasks)."""
    base_name = base_dataset_name(config)

    if base_name == "rotatedmnist":
        n_tasks = 10
        train_datasets, test_datasets, task_labels = (
            dataset.rotated_task_construction(n_tasks)
        )
        return train_datasets, test_datasets, task_labels, n_tasks

    all_labels = list(range(config["num_labels"]))
    task_labels = utils.define_task_labels(all_labels, config["n"])

    if base_name.startswith("cifar100"):
        task_labels = task_labels[:5]
    elif base_name.startswith("tinyimagenet"):
        task_labels = task_labels[:20]

    n_tasks = len(task_labels)
    class_il = bool(config.get("class_il", False))
    train_datasets, test_datasets = dataset.split_task_construction(
        base_name, task_labels, class_il=class_il
    )
    return train_datasets, test_datasets, task_labels, n_tasks


def apply_pred_strategy(model, config):
    """Sets model.pred_with_transformer from config['training_layers'].

    Returns (train_scope_summary, resolved_tokens).
    """
    training_layers = config.get("training_layers", None)
    tokens = resolve_training_layer_tokens(model, training_layers)
    model.pred_with_transformer = tokens
    train_scope = utils.summarize_train_scope(model, include_tokens=tokens)
    return train_scope, tokens


def resolve_training_layers(model, config):
    """Splits trainable parameters into method-managed and Adam-managed groups.

    training_layers is None or 'all' (everything goes to the method) or a
    list of module names (only matches go to the method). Every listed name
    must match at least one parameter.
    """
    training_layers = config.get("training_layers", None)

    if training_layers is None or training_layers == "all":
        method_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        return method_params, []

    if training_layers == "all_but_head":
        training_layers = resolve_training_layer_tokens(model, "all_but_head")

    all_parameter_names = [name for name, _ in model.named_parameters()]
    unmatched = [
        token
        for token in training_layers
        if not any(
            utils._name_matches_any_token(name, [token])
            for name in all_parameter_names
        )
    ]
    if unmatched:
        raise ValueError(
            f"training_layers contains names that match no parameters "
            f"in {type(model).__name__}: {sorted(unmatched)}"
        )

    method_params = []
    adam_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if utils._name_matches_any_token(name, training_layers):
            method_params.append((name, param))
        else:
            adam_params.append((name, param))
    return method_params, adam_params


def make_adam_optimizer(adam_params, config):
    if not adam_params:
        return None
    return torch.optim.Adam(
        [param for _, param in adam_params],
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("lambda_l2", 0.0),
    )


def evaluate_on_tasks(model, test_datasets, up_to_task, device, batch_size):
    model.eval()
    results = {}
    for task_idx in range(up_to_task + 1):
        correct, total = _count_correct(
            model, test_datasets[task_idx], device, batch_size
        )
        results[task_idx] = round(correct / total, 4) if total > 0 else 0.0
    model.train()
    return results


def evaluate_future_tasks(
    model, test_datasets, current_task, n_tasks, device, batch_size
):
    model.eval()
    results = {}
    for task_idx in range(current_task + 1, n_tasks):
        correct, total = _count_correct(
            model, test_datasets[task_idx], device, batch_size
        )
        results[task_idx] = round(correct / total, 4) if total > 0 else 0.0
    model.train()
    return results


def _count_correct(model, test_ds, device, batch_size):
    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct, total


def steps_to_epochs(total_steps, dataset_size, batch_size):
    batches_per_epoch = max(1, math.ceil(dataset_size / batch_size))
    return max(1, round(total_steps / batches_per_epoch))


def compute_equivalent_epochs(config, train_dataset_size):
    """Returns (base_epochs, warmup_epochs) from explicit epochs or steps."""
    if config.get("train_epochs") is not None:
        return config["train_epochs"], config.get("warmup_epochs", 0)

    batch_size = config.get("train_mb_size", config.get("batch_size", 32))
    base_epochs = steps_to_epochs(
        config["steps"], train_dataset_size, batch_size
    )
    warmup_epochs = steps_to_epochs(
        config.get("warmup_steps", 0), train_dataset_size, batch_size
    )
    return base_epochs, warmup_epochs


def profile_summary(task_profiles):
    if not task_profiles:
        return {}

    gpu_alloc = [v["gpu_peak_allocated_mb"] for v in task_profiles.values()]
    gpu_res = [v["gpu_peak_reserved_mb"] for v in task_profiles.values()]
    cpu_delta = [v["cpu_delta_mb"] for v in task_profiles.values()]
    time_s = [v["time_s"] for v in task_profiles.values()]

    def stats(values, prefix):
        return {
            f"{prefix}_mean": round(sum(values) / len(values), 2),
            f"{prefix}_max": round(max(values), 2),
            f"{prefix}_min": round(min(values), 2),
        }

    return {
        **stats(gpu_alloc, "gpu_peak_allocated_mb"),
        **stats(gpu_res, "gpu_peak_reserved_mb"),
        **stats(cpu_delta, "cpu_delta_mb"),
        **stats(time_s, "fit_time_s"),
    }


RESULTS_COMMON_KEYS = [
    "dataset",
    "base_dataset",
    "method",
    "model_class",
    "seed",
    "training_layers",
    "n_tasks",
    "steps",
    "warmup_steps",
    "test_steps",
    "train_epochs",
    "warmup_epochs",
    "lr",
    "base_params",
    "transformer_params",
    "selected_nweights",
    "selected_nlayers",
    "training_scope_label",
    "trained_modules",
]


def build_results_row(
    config,
    method_name,
    metrics,
    train_scope,
    task_profiles,
    base_params,
    transformer_params=0,
    extra=None,
):
    row = {
        "dataset": config["dataset"],
        "base_dataset": base_dataset_name(config),
        "method": method_name,
        "model_class": config["model_class"],
        "seed": config["seed"],
        "training_layers": _training_layers_label(config.get("training_layers")),
        "n_tasks": metrics.get("n_tasks"),
        "steps": config.get("steps"),
        "warmup_steps": config.get("warmup_steps"),
        "test_steps": config.get("test_steps"),
        "train_epochs": config.get("train_epochs"),
        "warmup_epochs": config.get("warmup_epochs"),
        "lr": config.get("baseline_lr", config.get("lr")),
        "base_params": int(base_params) if base_params is not None else None,
        "transformer_params": int(transformer_params),
        "selected_nweights": train_scope.get("selected_total_parameters"),
        "selected_nlayers": train_scope.get("selected_parameter_tensors"),
        "training_scope_label": train_scope.get("training_scope_label"),
        "trained_modules": train_scope.get("trainable_leaf_module_names"),
    }
    if extra:
        row.update(extra)
    row.update(metrics)
    row.update(profile_summary(task_profiles))
    return row


def _training_layers_label(training_layers):
    if training_layers is None:
        return "all"
    if isinstance(training_layers, str):
        return training_layers
    return ",".join(training_layers)
