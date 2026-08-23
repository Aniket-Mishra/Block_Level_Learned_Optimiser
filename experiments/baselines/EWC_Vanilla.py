"""Vanilla EWC baseline handler.

After each task it stores that task's diagonal Fisher and a parameter
snapshot; the penalty sums an independent quadratic term over every past
task, so memory grows linearly with the number of tasks (unlike the online
variant, which keeps one running Fisher and one snapshot).

Reference:
    Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
    networks", PNAS 2017.
"""

import random
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from models import base_models
from profiler import Profiler
from baselines.baseline_handler import (
    load_split_datasets,
    evaluate_on_tasks,
    evaluate_future_tasks,
    apply_pred_strategy,
    resolve_training_layers,
    make_adam_optimizer,
    compute_equivalent_epochs,
    build_results_row,
)


def compute_fisher(model, train_ds, device, batch_size, method_param_names):
    model.eval()
    fisher = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if name in method_param_names
    }

    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    n_samples = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        for name, param in model.named_parameters():
            if name in fisher and param.grad is not None:
                fisher[name] += param.grad.data.pow(2) * x.size(0)
        n_samples += x.size(0)

    for name in fisher:
        fisher[name] /= max(n_samples, 1)

    model.train()
    return fisher


def ewc_penalty(model, ewc_tasks, method_param_names):
    """Sums the quadratic penalty over every stored past task."""
    penalty = 0.0
    for name, param in model.named_parameters():
        if name not in method_param_names:
            continue
        for past_task in ewc_tasks:
            fisher = past_task["fisher"]
            if name in fisher:
                drift = (param - past_task["params"][name]).pow(2)
                penalty += (fisher[name] * drift).sum()
    return penalty


def train_one_epoch(model, train_ds, optimizer, adam_optimizer, criterion,
                    device, batch_size, ewc_lambda, ewc_tasks,
                    method_param_names):
    model.train()
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if adam_optimizer is not None:
            adam_optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        if ewc_tasks:
            loss = loss + (ewc_lambda / 2.0) * ewc_penalty(
                model, ewc_tasks, method_param_names
            )

        loss.backward()
        optimizer.step()
        if adam_optimizer is not None:
            adam_optimizer.step()


def handler(args):
    config = vars(args)
    ds_name = config["dataset"]
    print(f"\n##### Baseline: ewc_vanilla on {ds_name}")

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_datasets, test_datasets, task_labels, n_tasks = (
        load_split_datasets(config)
    )

    model = base_models.__dict__[config["model_class"]](
        config, n_heads=None, n_classes=config["n"]
    ).to(device)
    train_scope, scope_tokens = apply_pred_strategy(model, config)

    method_params, adam_params = resolve_training_layers(model, config)
    method_param_names = {n for n, _ in method_params}
    adam_optimizer = make_adam_optimizer(adam_params, config)

    criterion = nn.CrossEntropyLoss()
    baseline_lr = config.get("baseline_lr", config.get("lr", 1e-3))
    optimizer = torch.optim.Adam(
        [p for _, p in method_params], lr=baseline_lr
    )

    ewc_lambda = config.get("ewc_lambda", 400.0)
    base_params = sum(p.numel() for p in model.parameters())

    ewc_tasks = []
    previous_tasks_acc = {}
    future_tasks_acc = {}
    task_times = {}
    task_profiles = {}
    start_time = time.time()

    base_epochs, warmup_epochs = compute_equivalent_epochs(
        config, len(train_datasets[0])
    )

    for task_idx in range(n_tasks):
        epochs_this_task = (
            base_epochs + warmup_epochs if task_idx == 0 else base_epochs
        )
        print(f"Task {task_idx}/{n_tasks} (epochs={epochs_this_task})")
        task_start = time.time()

        with Profiler(label=f"task_{task_idx}") as p:
            for epoch in tqdm(range(epochs_this_task), desc=f"task {task_idx}", leave=False):
                train_one_epoch(
                    model, train_datasets[task_idx], optimizer,
                    adam_optimizer, criterion, device,
                    config["train_mb_size"], ewc_lambda,
                    ewc_tasks, method_param_names,
                )

            fisher = compute_fisher(
                model, train_datasets[task_idx], device,
                config["eval_mb_size"], method_param_names,
            )
            saved_params = {
                n: p.data.clone()
                for n, p in model.named_parameters()
                if n in method_param_names
            }
            ewc_tasks.append({"fisher": fisher, "params": saved_params})

        task_profiles[task_idx] = p.result.summary()

        previous_tasks_acc[task_idx] = evaluate_on_tasks(
            model, test_datasets, task_idx, device, config["eval_mb_size"]
        )
        if task_idx + 1 < n_tasks:
            future_tasks_acc[task_idx] = evaluate_future_tasks(
                model, test_datasets, task_idx, n_tasks, device,
                config["eval_mb_size"]
            )

        task_times[task_idx] = round(time.time() - task_start, 2)

    total_time = time.time() - start_time

    metrics = utils.compute_all_metrics(
        previous_tasks_acc=previous_tasks_acc,
        future_tasks_acc=future_tasks_acc,
        task_times=task_times,
        n_tasks=n_tasks,
        total_time=total_time,
        base_params=base_params,
        transformer_params=0,
    )

    print(f"avg_acc_final={metrics['average_accuracy_final']}")
    print(f"bwt={metrics['bwt_avg_over_time']}")
    print(f"fwt={metrics['mean_future_accuracy_over_time']}")

    return build_results_row(
        config, "ewc_vanilla", metrics, train_scope, task_profiles,
        base_params, extra={
            "ewc_lambda": ewc_lambda,
            "n_tasks": n_tasks,
        }
    )
