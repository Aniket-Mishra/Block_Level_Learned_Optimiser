"""La-MAML (Look-ahead MAML) baseline handler.

Reimplemented around the shared base models and data pipeline. Keeps
per-parameter learnable learning rates; for each batch it fast-adapts the
weights, evaluates the adapted weights on the replay buffer, then updates
both the weights and the learning rates from that meta loss.

Reference:
    Gupta et al., NeurIPS 2020. https://arxiv.org/abs/2007.13904
"""

import copy
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
    apply_pred_strategy,
    build_results_row,
)


class EpisodicMemory:
    """Reservoir-sampled (x, y) buffer: a full buffer replaces entries with
    probability mem_size / n_seen, keeping a uniform sample over all data
    seen so far."""

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


class LaMAMLTrainer:
    """Wraps a base model with per-parameter learning rates and the La-MAML
    meta-update rule."""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        self.alpha_lr = nn.ParameterList([
            nn.Parameter(
                torch.full_like(p, config["inner_lr_init"])
            )
            for p in model.parameters() if p.requires_grad
        ])
        self.alpha_lr.to(device)

        self.opt_wt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.get("meta_lr", config["lr"])
        )
        self.opt_lr = torch.optim.Adam(
            self.alpha_lr.parameters(), lr=config["alpha_lr_lr"]
        )

        self.memory = EpisodicMemory(config["mem_size"], device)
        self.grad_clip = config["grad_clip_norm"]

    def _trainable_params(self):
        """Trainable parameters, in the same order as alpha_lr."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def inner_update(self, x, y, fast_weights):
        """One fast update: w' = w - relu(alpha) * clamp(grad)."""
        if fast_weights is None:
            fast_weights = self._trainable_params()

        # Forward with fast_weights by temporarily swapping param data.
        originals = []
        params = self._trainable_params()
        for p, fw in zip(params, fast_weights):
            originals.append(p.data.clone())
            p.data = fw.data if isinstance(fw, nn.Parameter) else fw

        logits = self.model(x)
        loss = self.criterion(logits, y)

        grads = torch.autograd.grad(
            loss, params,
            create_graph=self.config["second_order"],
            retain_graph=self.config["second_order"],
            allow_unused=True,
        )

        for p, orig in zip(params, originals):
            p.data = orig

        new_fast_weights = []
        for i, (fw, g) in enumerate(zip(fast_weights, grads)):
            if g is None:
                new_fast_weights.append(fw)
            else:
                g = torch.clamp(g, -self.grad_clip, self.grad_clip)
                new_fast_weights.append(
                    fw - F.relu(self.alpha_lr[i]) * g
                )
        return new_fast_weights

    def meta_loss(self, x, y, fast_weights):
        """Loss of the adapted weights on (x, y), typically replay data."""
        params = self._trainable_params()
        originals = []
        for p, fw in zip(params, fast_weights):
            originals.append(p.data.clone())
            p.data = fw.data if isinstance(fw, nn.Parameter) else fw

        logits = self.model(x)
        loss = self.criterion(logits, y)

        for p, orig in zip(params, originals):
            p.data = orig

        return loss

    def observe(self, x, y):
        """One batch: inner updates on current data, meta update on replay.
        Returns the meta loss value for logging."""
        self.model.train()

        for _ in range(self.config["glances"]):
            perm = torch.randperm(x.size(0))
            x, y = x[perm], y[perm]

            if len(self.memory) > 0:
                bx, by = self.memory.sample(x.size(0))
            else:
                bx, by = x, y

            # Inner updates process samples one at a time.
            fast_weights = None
            for i in range(x.size(0)):
                xi = x[i].unsqueeze(0)
                yi = y[i].unsqueeze(0)
                fast_weights = self.inner_update(xi, yi, fast_weights)

            meta_loss = self.meta_loss(bx, by, fast_weights)

            self.opt_wt.zero_grad()
            self.opt_lr.zero_grad()
            meta_loss.backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            nn.utils.clip_grad_norm_(
                self.alpha_lr.parameters(), self.grad_clip
            )

            self.opt_lr.step()
            self.opt_wt.step()

        self.memory.push(x, y)

        return meta_loss.item()

    def train_on_task(self, train_ds, n_epochs, batch_size):
        loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True,
        )
        for epoch in tqdm(range(n_epochs), desc="  epochs", leave=False):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                self.observe(x, y)

    def adapt_and_evaluate(self, test_ds, test_steps, batch_size):
        """Test-time adaptation matching the proposed method's test_steps:
        deep-copies the model, runs test_steps update steps on shuffled test
        batches, then evaluates on the full test set. The original model is
        not modified."""
        if test_steps <= 0:
            return self._evaluate_no_adapt(test_ds, batch_size)

        loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True,
        )

        eval_model = copy.deepcopy(self.model)
        eval_alpha = copy.deepcopy(self.alpha_lr)
        eval_model.eval()

        eval_opt = torch.optim.SGD(
            eval_model.parameters(), lr=self.config.get("meta_lr", self.config["lr"])
        )

        eval_model.train()
        steps_done = 0
        for x, y in loader:
            if steps_done >= test_steps:
                break
            x, y = x.to(self.device), y.to(self.device)
            eval_opt.zero_grad()
            logits = eval_model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(
                eval_model.parameters(), self.grad_clip
            )
            eval_opt.step()
            steps_done += 1

        eval_model.eval()
        correct, total = 0, 0
        eval_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = eval_model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        del eval_model, eval_alpha, eval_opt
        return correct / total if total > 0 else 0.0

    def _evaluate_no_adapt(self, test_ds, batch_size):
        self.model.eval()
        loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        self.model.train()
        return correct / total if total > 0 else 0.0


def handler(args):
    config = vars(args)
    ds_name = config["dataset"]
    print(f"\n##### Baseline: lamaml on {ds_name}")

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

    base_params = sum(p.numel() for p in model.parameters())

    trainer = LaMAMLTrainer(model, config, device)
    alpha_params = sum(p.numel() for p in trainer.alpha_lr.parameters())

    previous_tasks_acc = {}
    future_tasks_acc = {}
    task_times = {}
    task_profiles = {}
    start_time = time.time()

    warmup_epochs = config.get("warmup_epochs", 0)
    base_epochs = config["train_epochs"]
    test_steps = config.get("test_steps", 0)

    for task_idx in range(n_tasks):
        epochs_this_task = (
            base_epochs + warmup_epochs if task_idx == 0 else base_epochs
        )
        print(f"Task {task_idx}/{n_tasks} (epochs={epochs_this_task})")
        task_start = time.time()

        with Profiler(label=f"task_{task_idx}") as p:
            trainer.train_on_task(
                train_datasets[task_idx],
                n_epochs=epochs_this_task,
                batch_size=config["train_mb_size"],
            )
        task_profiles[task_idx] = p.result.summary()

        # Previous tasks: with test-time adaptation. Future tasks: without
        # (the task is unseen).
        prev_acc = {}
        for t in range(task_idx + 1):
            acc = trainer.adapt_and_evaluate(
                test_datasets[t], test_steps,
                batch_size=config["eval_mb_size"],
            )
            prev_acc[t] = round(acc, 4)
        previous_tasks_acc[task_idx] = prev_acc

        if task_idx + 1 < n_tasks:
            future_acc = {}
            for t in range(task_idx + 1, n_tasks):
                acc = trainer._evaluate_no_adapt(
                    test_datasets[t],
                    batch_size=config["eval_mb_size"],
                )
                future_acc[t] = round(acc, 4)
            future_tasks_acc[task_idx] = future_acc

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
        config, "lamaml", metrics, train_scope, task_profiles,
        base_params, extra={
            "alpha_params": alpha_params,
            "n_tasks": n_tasks,
            "mem_size": config["mem_size"],
            "inner_lr_init": config["inner_lr_init"],
            "test_steps": test_steps,
        }
    )
