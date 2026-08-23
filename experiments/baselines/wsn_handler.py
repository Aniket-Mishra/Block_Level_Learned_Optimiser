"""WSN (Winning SubNetworks) baseline handler.

Reimplemented around the shared base models and data pipeline instead of
WSN's custom Subnet layers. Each task selects its top-c% weights by magnitude
as a binary mask; gradients for weights locked by any prior mask are zeroed,
and test time applies the task mask, giving the forget-free property.

Reference:
    Kang et al., ICML 2022. https://github.com/ihaeyong/WSN
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
    apply_pred_strategy,
    build_results_row,
)


def compute_task_mask(model, sparsity):
    """Top-c% weights by absolute magnitude per parameter tensor, as
    {param_name: bool_tensor}. sparsity is the fraction kept."""
    mask = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        flat = param.data.abs().flatten()
        k = max(1, int(len(flat) * sparsity))
        threshold = flat.topk(k).values[-1]
        mask[name] = (param.data.abs() >= threshold)
    return mask


def consolidate_masks(prior_mask, new_mask):
    """ORs a new task mask into the consolidated mask, keeping weights used
    by any prior task protected."""
    if prior_mask is None:
        return {k: v.clone() for k, v in new_mask.items()}
    merged = {}
    for k in new_mask:
        if k in prior_mask:
            merged[k] = prior_mask[k] | new_mask[k]
        else:
            merged[k] = new_mask[k].clone()
    return merged


def zero_consolidated_grads(model, consolidated_mask):
    if consolidated_mask is None:
        return
    for name, param in model.named_parameters():
        if name in consolidated_mask and param.grad is not None:
            param.grad[consolidated_mask[name]] = 0.0


def train_one_epoch(model, train_ds, optimizer, criterion, device,
                    batch_size, consolidated_mask):
    model.train()
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        zero_consolidated_grads(model, consolidated_mask)
        optimizer.step()


def evaluate_with_mask(model, test_ds, task_mask, device, batch_size):
    """Evaluates with only the task's subnetwork active: weights outside the
    mask are zeroed for the forward passes, then restored."""
    model.eval()
    originals = {}
    if task_mask is not None:
        for name, param in model.named_parameters():
            if name in task_mask:
                originals[name] = param.data.clone()
                param.data *= task_mask[name].float()

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

    params_by_name = dict(model.named_parameters())
    for name, data in originals.items():
        params_by_name[name].data.copy_(data)

    model.train()
    return correct, total


def evaluate_wsn_on_tasks(model, test_datasets, per_task_masks,
                          up_to_task, device, batch_size):
    results = {}
    for t in range(up_to_task + 1):
        mask_t = per_task_masks.get(t)
        correct, total = evaluate_with_mask(
            model, test_datasets[t], mask_t, device, batch_size
        )
        results[t] = round(correct / total, 4) if total > 0 else 0.0
    return results


def evaluate_wsn_future_tasks(model, test_datasets, current_task,
                              n_tasks, device, batch_size):
    """Future tasks have no mask yet, so the full model is evaluated: the
    forward-transfer measurement on unseen tasks."""
    results = {}
    for t in range(current_task + 1, n_tasks):
        loader = DataLoader(
            test_datasets[t], batch_size=batch_size, shuffle=False,
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
        results[t] = round(correct / total, 4) if total > 0 else 0.0
    return results


def handler(args):
    config = vars(args)
    ds_name = config["dataset"]
    print(f"\n##### Baseline: wsn on {ds_name}")

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

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params, lr=config.get("baseline_lr", config.get("lr", 1e-3))
    )
    sparsity = config["sparsity"]

    base_params = sum(p.numel() for p in model.parameters())

    consolidated_mask = None
    per_task_masks = {}
    previous_tasks_acc = {}
    future_tasks_acc = {}
    task_times = {}
    task_profiles = {}
    start_time = time.time()

    warmup_epochs = config.get("warmup_epochs", 0)
    base_epochs = config["train_epochs"]

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
                    criterion, device, config["train_mb_size"],
                    consolidated_mask,
                )
            task_mask = compute_task_mask(model, sparsity)
            per_task_masks[task_idx] = task_mask
            consolidated_mask = consolidate_masks(
                consolidated_mask, task_mask
            )

        task_profiles[task_idx] = p.result.summary()

        previous_tasks_acc[task_idx] = evaluate_wsn_on_tasks(
            model, test_datasets, per_task_masks, task_idx, device,
            batch_size=config["eval_mb_size"],
        )

        if task_idx + 1 < n_tasks:
            future_tasks_acc[task_idx] = evaluate_wsn_future_tasks(
                model, test_datasets, task_idx, n_tasks, device,
                batch_size=config["eval_mb_size"],
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
        config, "wsn", metrics, train_scope, task_profiles,
        base_params, extra={
            "sparsity": sparsity,
            "n_tasks": n_tasks,
        }
    )
