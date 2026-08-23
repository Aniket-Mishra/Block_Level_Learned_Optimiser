"""DER++ (Dark Experience Replay++) baseline handler.

Replay buffer of (x, y, logits). Loss = CE(current) + alpha * MSE(buffer
logits) + beta * CE(buffer labels).

Reference:
    Buzzega et al., "Dark Experience for General Continual Learning:
    a Strong, Simple Baseline", NeurIPS 2020.
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
import torch.nn.functional as F
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


class ReplayBuffer:
    """Reservoir-sampled buffer of (x, y, logits) triples: a full buffer
    replaces entries with probability mem_size / n_seen, keeping a uniform
    sample over all data seen so far."""

    def __init__(self, mem_size, device):
        self.mem_size = mem_size
        self.device = device
        self.buffer_x = []
        self.buffer_y = []
        self.buffer_logits = []
        self.n_seen = 0

    def push(self, x, y, logits):
        for i in range(x.size(0)):
            self.n_seen += 1
            xi = x[i].detach().cpu()
            yi = y[i].detach().cpu()
            li = logits[i].detach().cpu()
            if len(self.buffer_x) < self.mem_size:
                self.buffer_x.append(xi)
                self.buffer_y.append(yi)
                self.buffer_logits.append(li)
            else:
                idx = random.randint(0, self.n_seen - 1)
                if idx < self.mem_size:
                    self.buffer_x[idx] = xi
                    self.buffer_y[idx] = yi
                    self.buffer_logits[idx] = li

    def sample(self, batch_size):
        n = min(batch_size, len(self.buffer_x))
        indices = random.sample(range(len(self.buffer_x)), n)
        bx = torch.stack([self.buffer_x[i] for i in indices]).to(self.device)
        by = torch.stack([self.buffer_y[i] for i in indices]).to(self.device)
        bl = torch.stack([self.buffer_logits[i] for i in indices]).to(self.device)
        return bx, by, bl

    def __len__(self):
        return len(self.buffer_x)


def train_one_epoch(model, train_ds, optimizer, adam_optimizer, criterion,
                    device, batch_size, alpha, beta, buffer):
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

        if len(buffer) > 0:
            buf_x, buf_y, buf_logits = buffer.sample(batch_size)

            buf_out = model(buf_x)
            loss_mse = F.mse_loss(buf_out, buf_logits)
            loss_ce = criterion(buf_out, buf_y)
            loss = loss + alpha * loss_mse + beta * loss_ce

        loss.backward()
        optimizer.step()
        if adam_optimizer is not None:
            adam_optimizer.step()

        buffer.push(x, y, logits.detach())


def handler(args):
    config = vars(args)
    ds_name = config["dataset"]
    print(f"\n##### Baseline: der_pp on {ds_name}")

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
    adam_optimizer = make_adam_optimizer(adam_params, config)

    criterion = nn.CrossEntropyLoss()
    baseline_lr = config.get("baseline_lr", config.get("lr", 1e-3))
    optimizer = torch.optim.Adam(
        [p for _, p in method_params], lr=baseline_lr
    )

    alpha = config.get("der_alpha", 0.3)
    beta = config.get("der_beta", 0.5)
    mem_size = config.get("mem_size", 500)
    base_params = sum(p.numel() for p in model.parameters())

    buffer = ReplayBuffer(mem_size, device)
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
                    config["train_mb_size"], alpha, beta, buffer,
                )

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
        config, "der_pp", metrics, train_scope, task_profiles,
        base_params, extra={
            "der_alpha": alpha,
            "der_beta": beta,
            "mem_size": mem_size,
            "n_tasks": n_tasks,
        }
    )
