"""DyTox (Dynamic Token Expansion) baseline handler.

A shared ViT encoder plus one learnable task token per task. A Task
Attention Block cross-attends the task tokens to the encoder output;
each task has its own head, and old heads freeze after their task.

Reference:
    Douillard et al., "DyTox: Transformers for Continual Learning with
    DYnamic TOken eXpansion", CVPR 2022.
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
    compute_equivalent_epochs,
    build_results_row,
)


class TaskAttentionBlock(nn.Module):
    """Cross-attention block where task tokens attend to the encoder output,
    optionally after self-attention among the task tokens so later tasks can
    use earlier task representations."""

    def __init__(self, embed_dim, num_heads, use_self_attn=True):
        super().__init__()
        self.use_self_attn = use_self_attn

        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(
                embed_dim, num_heads, batch_first=True
            )
            self.norm_sa = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm_ca = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, task_tokens, encoder_output):
        """task_tokens (batch, n_task_tokens, embed_dim), encoder_output
        (batch, seq_len, embed_dim) -> (batch, n_task_tokens, embed_dim)."""
        if self.use_self_attn and task_tokens.size(1) > 1:
            sa_out, _ = self.self_attn(
                task_tokens, task_tokens, task_tokens
            )
            task_tokens = self.norm_sa(task_tokens + sa_out)

        ca_out, _ = self.cross_attn(
            task_tokens, encoder_output, encoder_output
        )
        task_tokens = self.norm_ca(task_tokens + ca_out)

        ffn_out = self.ffn(task_tokens)
        task_tokens = self.norm_ffn(task_tokens + ffn_out)

        return task_tokens


class DyToxModel(nn.Module):
    """DyTox wrapper around a ViTSmall backbone: task tokens, the TAB, and
    per-task classifiers. add_task() appends a token and a head."""

    def __init__(self, backbone, embed_dim, classes_per_task, num_heads=6):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.classes_per_task = classes_per_task

        self.tab = TaskAttentionBlock(embed_dim, num_heads, use_self_attn=True)

        self.task_tokens = nn.ParameterList()
        self.classifiers = nn.ModuleList()

        self._current_task = -1

    def add_task(self):
        self._current_task += 1
        new_token = nn.Parameter(
            torch.randn(1, 1, self.embed_dim) * 0.02
        )
        self.task_tokens.append(new_token)

        new_head = nn.Linear(self.embed_dim, self.classes_per_task)
        self.classifiers.append(new_head)

    def freeze_old_classifiers(self):
        for i in range(len(self.classifiers) - 1):
            for param in self.classifiers[i].parameters():
                param.requires_grad = False

    def _get_encoder_output(self, x):
        patches = self.backbone.patch_embed(x).flatten(2).transpose(1, 2)
        batch_size = patches.size(0)
        cls_tokens = self.backbone.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)
        tokens = tokens + self.backbone.pos_embed
        encoder_output = self.backbone.encoder(tokens)
        encoder_output = self.backbone.norm(encoder_output)
        return encoder_output

    def forward(self, x, task_idx=None):
        """With task_idx, returns that task head's logits only; without,
        returns all heads' logits concatenated (class-IL evaluation)."""
        encoder_output = self._get_encoder_output(x)
        batch_size = x.size(0)

        n_tokens = len(self.task_tokens)
        all_tokens = torch.cat(
            [t.expand(batch_size, -1, -1) for t in self.task_tokens],
            dim=1
        )

        tab_output = self.tab(all_tokens, encoder_output)

        if task_idx is not None:
            task_repr = tab_output[:, task_idx, :]
            return self.classifiers[task_idx](task_repr)

        all_logits = []
        for i in range(n_tokens):
            task_repr = tab_output[:, i, :]
            all_logits.append(self.classifiers[i](task_repr))
        return torch.cat(all_logits, dim=1)


def train_one_epoch_dytox(dytox_model, train_ds, optimizer, device,
                          batch_size, task_idx, use_task_specific_head):
    dytox_model.train()
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if use_task_specific_head:
            logits = dytox_model(x, task_idx=task_idx)
        else:
            logits = dytox_model(x)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()


# Local evaluators: the shared ones cannot route per-task heads.
def evaluate_dytox(dytox_model, test_datasets, up_to_task, device,
                   batch_size, use_task_specific_head):
    dytox_model.eval()
    results = {}
    for task_idx in range(up_to_task + 1):
        loader = DataLoader(
            test_datasets[task_idx], batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                if use_task_specific_head:
                    logits = dytox_model(x, task_idx=task_idx)
                else:
                    logits = dytox_model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        results[task_idx] = round(correct / total, 4) if total > 0 else 0.0
    dytox_model.train()
    return results


def evaluate_dytox_future(dytox_model, test_datasets, current_task, n_tasks,
                          device, batch_size):
    """Future tasks have no task token yet, so the full model output (all
    current heads concatenated) is used: forward transfer to unseen
    class-IL tasks."""
    dytox_model.eval()
    results = {}
    for task_idx in range(current_task + 1, n_tasks):
        loader = DataLoader(
            test_datasets[task_idx], batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = dytox_model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        results[task_idx] = round(correct / total, 4) if total > 0 else 0.0
    dytox_model.train()
    return results


def handler(args):
    config = vars(args)
    ds_name = config["dataset"]
    print(f"\n##### Baseline: dytox on {ds_name}")

    if config["model_class"] != "ViTSmall":
        raise ValueError(
            "DyTox requires ViTSmall. Got model_class="
            f"'{config['model_class']}'"
        )

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_class_il = config.get("dytox_class_il", False)

    if use_class_il:
        config_load = dict(config)
        config_load["class_il"] = True
        train_datasets, test_datasets, task_labels, n_tasks = (
            load_split_datasets(config_load)
        )
        classes_per_task = config["n"]
    else:
        train_datasets, test_datasets, task_labels, n_tasks = (
            load_split_datasets(config)
        )
        classes_per_task = config["n"]

    backbone = base_models.ViTSmall(
        config, n_heads=None, n_classes=config["n"]
    ).to(device)

    embed_dim = config["features_dim"]
    tab_heads = config.get("dytox_tab_heads", 6)

    dytox_model = DyToxModel(
        backbone, embed_dim, classes_per_task, num_heads=tab_heads
    ).to(device)

    train_scope, scope_tokens = apply_pred_strategy(backbone, config)

    base_params = sum(p.numel() for p in backbone.parameters())

    previous_tasks_acc = {}
    future_tasks_acc = {}
    task_times = {}
    task_profiles = {}
    start_time = time.time()

    base_epochs, warmup_epochs = compute_equivalent_epochs(
        config, len(train_datasets[0])
    )

    for task_idx in range(n_tasks):
        dytox_model.add_task()
        dytox_model.to(device)
        dytox_model.freeze_old_classifiers()

        trainable_params = [
            p for p in dytox_model.parameters() if p.requires_grad
        ]
        baseline_lr = config.get("baseline_lr", config.get("lr", 1e-3))
        optimizer = torch.optim.Adam(trainable_params, lr=baseline_lr)

        epochs_this_task = (
            base_epochs + warmup_epochs if task_idx == 0 else base_epochs
        )
        print(f"Task {task_idx}/{n_tasks} (epochs={epochs_this_task})")
        task_start = time.time()

        with Profiler(label=f"task_{task_idx}") as p:
            for epoch in tqdm(range(epochs_this_task), desc=f"task {task_idx}", leave=False):
                train_one_epoch_dytox(
                    dytox_model, train_datasets[task_idx], optimizer,
                    device, config["train_mb_size"], task_idx,
                    use_task_specific_head=not use_class_il,
                )

        task_profiles[task_idx] = p.result.summary()

        previous_tasks_acc[task_idx] = evaluate_dytox(
            dytox_model, test_datasets, task_idx, device,
            config["eval_mb_size"],
            use_task_specific_head=not use_class_il,
        )
        if task_idx + 1 < n_tasks:
            future_tasks_acc[task_idx] = evaluate_dytox_future(
                dytox_model, test_datasets, task_idx, n_tasks, device,
                config["eval_mb_size"]
            )

        task_times[task_idx] = round(time.time() - task_start, 2)

    total_time = time.time() - start_time

    dytox_extra_params = (
        sum(p.numel() for p in dytox_model.tab.parameters())
        + sum(p.numel() for p in dytox_model.task_tokens)
        + sum(p.numel() for p in dytox_model.classifiers.parameters())
    )

    metrics = utils.compute_all_metrics(
        previous_tasks_acc=previous_tasks_acc,
        future_tasks_acc=future_tasks_acc,
        task_times=task_times,
        n_tasks=n_tasks,
        total_time=total_time,
        base_params=base_params,
        transformer_params=dytox_extra_params,
    )

    print(f"avg_acc_final={metrics['average_accuracy_final']}")
    print(f"bwt={metrics['bwt_avg_over_time']}")
    print(f"fwt={metrics['mean_future_accuracy_over_time']}")

    return build_results_row(
        config, "dytox", metrics, train_scope, task_profiles,
        base_params, extra={
            "dytox_extra_params": dytox_extra_params,
            "dytox_tab_heads": tab_heads,
            "dytox_class_il": use_class_il,
            "n_tasks": n_tasks,
        }
    )
