"""ER-ACE (Experience Replay with Asymmetric Cross-Entropy) baseline handler.

Runs class-incremental. Current-task samples use cross-entropy masked to the
current classes; replay samples use cross-entropy over all seen classes.
That asymmetry is the key idea.

Reference:
    Caccia et al., "New Insights on Reducing Abrupt Representation Change
    in Online Continual Learning", ICLR 2022.
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


class ReplayBufferXY:
    """Reservoir-sampled buffer storing (x, y) pairs."""

    def __init__(self, mem_size, device):
        self.mem_size = mem_size
        self.device = device
        self.buffer_x = []
        self.buffer_y = []
        self.n_seen = 0

    def push(self, x, y):
        for i in range(x.size(0)):
            self.n_seen += 1
            if len(self.buffer_x) < self.mem_size:
                self.buffer_x.append(x[i].detach().cpu())
                self.buffer_y.append(y[i].detach().cpu())
            else:
                idx = random.randint(0, self.n_seen - 1)
                if idx < self.mem_size:
                    self.buffer_x[idx] = x[i].detach().cpu()
                    self.buffer_y[idx] = y[i].detach().cpu()

    def sample(self, batch_size):
        n = min(batch_size, len(self.buffer_x))
        indices = random.sample(range(len(self.buffer_x)), n)
        bx = torch.stack([self.buffer_x[i] for i in indices]).to(self.device)
        by = torch.stack([self.buffer_y[i] for i in indices]).to(self.device)
        return bx, by

    def __len__(self):
        return len(self.buffer_x)


def masked_cross_entropy(logits, targets, allowed_classes, num_classes):
    """Cross-entropy over allowed_classes only: other logits get -inf so they
    contribute zero probability in the softmax denominator."""
    mask = torch.full((num_classes,), float("-inf"), device=logits.device)
    mask[list(allowed_classes)] = 0.0
    masked_logits = logits + mask.unsqueeze(0)
    return nn.functional.cross_entropy(masked_logits, targets)


def train_one_epoch(model, train_ds, optimizer, adam_optimizer, device,
                    batch_size, current_classes, seen_classes, num_classes,
                    buffer):
    model.train()
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if adam_optimizer is not None:
            adam_optimizer.zero_grad()

        logits = model(x)
        loss = masked_cross_entropy(
            logits, y, current_classes, num_classes
        )

        if len(buffer) > 0:
            buf_x, buf_y = buffer.sample(batch_size)
            buf_logits = model(buf_x)
            replay_loss = masked_cross_entropy(
                buf_logits, buf_y, seen_classes, num_classes
            )
            loss = loss + replay_loss

        loss.backward()
        optimizer.step()
        if adam_optimizer is not None:
            adam_optimizer.step()

        buffer.push(x, y)


def handler(args):
    config = vars(args)
    ds_name = config["dataset"]
    print(f"\n##### Baseline: er_ace on {ds_name}")

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_class_il = dict(config)
    config_class_il["class_il"] = True
    train_datasets, test_datasets, task_labels, n_tasks = (
        load_split_datasets(config_class_il)
    )

    num_classes = config["num_labels"]
    model_config = dict(config)
    model_config["n"] = num_classes
    model = base_models.__dict__[config["model_class"]](
        model_config, n_heads=None, n_classes=num_classes
    ).to(device)
    train_scope, scope_tokens = apply_pred_strategy(model, config)

    method_params, adam_params = resolve_training_layers(model, config)
    adam_optimizer = make_adam_optimizer(adam_params, config)

    baseline_lr = config.get("baseline_lr", config.get("lr", 1e-3))
    optimizer = torch.optim.Adam(
        [p for _, p in method_params], lr=baseline_lr
    )

    mem_size = config.get("mem_size", 500)
    base_params = sum(p.numel() for p in model.parameters())

    buffer = ReplayBufferXY(mem_size, device)
    seen_classes = set()
    previous_tasks_acc = {}
    future_tasks_acc = {}
    task_times = {}
    task_profiles = {}
    start_time = time.time()

    base_epochs, warmup_epochs = compute_equivalent_epochs(
        config, len(train_datasets[0])
    )

    for task_idx in range(n_tasks):
        if isinstance(task_labels[task_idx], (list, tuple)):
            current_classes = set(int(c) for c in task_labels[task_idx])
        else:
            current_classes = set(range(num_classes))
        seen_classes = seen_classes | current_classes

        epochs_this_task = (
            base_epochs + warmup_epochs if task_idx == 0 else base_epochs
        )
        print(f"Task {task_idx}/{n_tasks} (epochs={epochs_this_task})")
        task_start = time.time()

        with Profiler(label=f"task_{task_idx}") as p:
            for epoch in tqdm(range(epochs_this_task), desc=f"task {task_idx}", leave=False):
                train_one_epoch(
                    model, train_datasets[task_idx], optimizer,
                    adam_optimizer, device, config["train_mb_size"],
                    current_classes, seen_classes, num_classes, buffer,
                )

        task_profiles[task_idx] = p.result.summary()

        # Class-IL evaluation: each task keeps its original (unmapped) labels,
        # so the shared evaluators apply directly.
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
        config, "er_ace", metrics, train_scope, task_profiles,
        base_params, extra={
            "mem_size": mem_size,
            "n_tasks": n_tasks,
            "class_il": True,
        }
    )
