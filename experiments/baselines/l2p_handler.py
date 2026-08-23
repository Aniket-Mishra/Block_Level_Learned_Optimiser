"""L2P (Learning to Prompt) baseline handler.

A frozen ViT backbone with a pool of learnable prompts selected per input by
key similarity; only the prompt pool, keys, and head train. Shallow prompting
only (prompts at the input layer). With a randomly initialised ViTSmall the
frozen features are weak, so accuracy is limited; load pretrained weights
before freezing if available.

Reference:
    Wang et al., "Learning to Prompt for Continual Learning", CVPR 2022.
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
    apply_pred_strategy,
    compute_equivalent_epochs,
    build_results_row,
)


class PromptPool(nn.Module):
    """Pool of learnable prompts with key-based selection: pool_size prompts
    of prompt_length tokens each, top_k selected per input."""

    def __init__(self, pool_size, prompt_length, embed_dim, top_k):
        super().__init__()
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.top_k = top_k

        self.prompts = nn.Parameter(
            torch.randn(pool_size, prompt_length, embed_dim) * 0.02
        )
        self.keys = nn.Parameter(
            torch.randn(pool_size, embed_dim) * 0.02
        )

    def forward(self, query):
        """Selects top_k prompts per input by cosine similarity.

        query: (batch, embed_dim), typically the mean patch embedding.
        Returns (selected_prompts of shape (batch, top_k * prompt_length,
        embed_dim), similarity_loss encouraging diverse prompt usage).
        """
        query_norm = F.normalize(query, dim=-1)
        keys_norm = F.normalize(self.keys, dim=-1)
        similarity = torch.matmul(query_norm, keys_norm.T)

        topk_values, topk_indices = similarity.topk(self.top_k, dim=-1)

        # Batched gather; same values as selecting per sample in a loop.
        selected_prompts = self.prompts[topk_indices].reshape(
            query.size(0), -1, self.embed_dim
        )

        similarity_loss = -topk_values.mean()
        return selected_prompts, similarity_loss


class L2PModel(nn.Module):
    """ViTSmall backbone (frozen) plus a prompt pool; only the pool, keys,
    and classifier head train."""

    def __init__(self, backbone, prompt_pool, num_classes):
        super().__init__()
        self.backbone = backbone
        self.prompt_pool = prompt_pool
        embed_dim = prompt_pool.embed_dim

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        patches = self.backbone.patch_embed(x).flatten(2).transpose(1, 2)
        batch_size = patches.size(0)

        query = patches.mean(dim=1)
        selected_prompts, sim_loss = self.prompt_pool(query)

        cls_tokens = self.backbone.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, selected_prompts, patches], dim=1)

        num_prompt_tokens = selected_prompts.size(1)
        num_patch_tokens = patches.size(1)
        total_tokens = 1 + num_prompt_tokens + num_patch_tokens

        # Prompt positions get zero positional embeddings.
        pos_embed = self.backbone.pos_embed
        orig_len = pos_embed.size(1)
        if total_tokens != orig_len:
            cls_pos = pos_embed[:, :1, :]
            patch_pos = pos_embed[:, 1:, :]
            prompt_pos = torch.zeros(
                1, num_prompt_tokens, pos_embed.size(2),
                device=pos_embed.device
            )
            pos_embed = torch.cat([cls_pos, prompt_pos, patch_pos], dim=1)

        tokens = tokens + pos_embed
        tokens = self.backbone.encoder(tokens)
        tokens = self.backbone.norm(tokens[:, 0])
        logits = self.classifier(tokens)
        return logits, sim_loss


def train_one_epoch(l2p_model, train_ds, optimizer, device, batch_size,
                    pull_weight):
    l2p_model.train()
    l2p_model.backbone.eval()
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, sim_loss = l2p_model(x)
        loss = criterion(logits, y) + pull_weight * sim_loss
        loss.backward()
        optimizer.step()


# Local evaluators: the model returns (logits, sim_loss) tuples, which the
# shared evaluators do not handle.
def evaluate_l2p(l2p_model, test_datasets, up_to_task, device, batch_size):
    l2p_model.eval()
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
                logits, _ = l2p_model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        results[task_idx] = round(correct / total, 4) if total > 0 else 0.0
    l2p_model.train()
    return results


def evaluate_l2p_future(l2p_model, test_datasets, current_task, n_tasks,
                        device, batch_size):
    l2p_model.eval()
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
                logits, _ = l2p_model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        results[task_idx] = round(correct / total, 4) if total > 0 else 0.0
    l2p_model.train()
    return results


def handler(args):
    config = vars(args)
    ds_name = config["dataset"]
    print(f"\n##### Baseline: l2p on {ds_name}")

    if config["model_class"] != "ViTSmall":
        raise ValueError(
            "L2P requires ViTSmall. Got model_class="
            f"'{config['model_class']}'"
        )

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_datasets, test_datasets, task_labels, n_tasks = (
        load_split_datasets(config)
    )

    backbone = base_models.ViTSmall(
        config, n_heads=None, n_classes=config["n"]
    ).to(device)

    embed_dim = config["features_dim"]
    pool_size = config.get("l2p_pool_size", 10)
    prompt_length = config.get("l2p_prompt_length", 5)
    top_k = config.get("l2p_top_k", 5)
    pull_weight = config.get("l2p_pull_weight", 0.1)

    prompt_pool = PromptPool(pool_size, prompt_length, embed_dim, top_k)
    l2p_model = L2PModel(backbone, prompt_pool, config["n"]).to(device)

    # Backbone is frozen, so the scope is taken over the whole wrapper:
    # L2P trains the prompt pool, the keys, and the classifier head.
    train_scope, scope_tokens = apply_pred_strategy(l2p_model, config)

    trainable_params = [
        p for p in l2p_model.parameters() if p.requires_grad
    ]
    baseline_lr = config.get("baseline_lr", config.get("lr", 1e-3))
    optimizer = torch.optim.Adam(trainable_params, lr=baseline_lr)

    base_params = sum(p.numel() for p in backbone.parameters())
    prompt_params = sum(p.numel() for p in prompt_pool.parameters())
    classifier_params = sum(
        p.numel() for p in l2p_model.classifier.parameters()
    )

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
                    l2p_model, train_datasets[task_idx], optimizer,
                    device, config["train_mb_size"], pull_weight,
                )

        task_profiles[task_idx] = p.result.summary()

        previous_tasks_acc[task_idx] = evaluate_l2p(
            l2p_model, test_datasets, task_idx, device,
            config["eval_mb_size"]
        )
        if task_idx + 1 < n_tasks:
            future_tasks_acc[task_idx] = evaluate_l2p_future(
                l2p_model, test_datasets, task_idx, n_tasks, device,
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
        transformer_params=prompt_params + classifier_params,
    )

    print(f"avg_acc_final={metrics['average_accuracy_final']}")
    print(f"bwt={metrics['bwt_avg_over_time']}")
    print(f"fwt={metrics['mean_future_accuracy_over_time']}")

    return build_results_row(
        config, "l2p", metrics, train_scope, task_profiles,
        base_params, extra={
            "l2p_pool_size": pool_size,
            "l2p_prompt_length": prompt_length,
            "l2p_top_k": top_k,
            "l2p_pull_weight": pull_weight,
            "prompt_params": prompt_params,
            "classifier_params": classifier_params,
            "n_tasks": n_tasks,
        }
    )
