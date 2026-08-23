"""Base-paper L2L (learning-to-learn) baseline handler.

Runs the original meta-optimizer from the base paper (baselines/l2l_base)/

Reference:
    Vettoruzzo et al., the base paper this thesis extends.
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

import utils
from models import base_models
from profiler import Profiler
from baselines.baseline_handler import (
    load_split_datasets,
    apply_pred_strategy,
    build_results_row,
)
from baselines.l2l_base import l2l_transformer, l2l_utils
from baselines.l2l_base.l2l_method import PROPOSED as L2LMethod


def _svhn_task_encoder_path():
    """Path to the SVHN-pretrained MNIST task encoder.

    The base paper always loads this checkpoint for SplitMNIST and
    RotatedMNIST; it is the same file the proposed method loads in
    exp_runner_1_1_fn_setup._build_task_encoder.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "pretrained_task_encoders"
        / "task_encoder_svhn.pth"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"SVHN task-encoder checkpoint not found: {path}. The base paper "
            "loads an SVHN-pretrained task encoder for the MNIST datasets, "
            "so the L2L baseline requires the same checkpoint the proposed "
            "method uses."
        )
    return str(path)


def _managed_conv_filters(model):
    """Total out*in over managed 4D conv weights (one transformer token each)."""
    filters = 0
    for name, param in model.named_parameters():
        managed = any(token in name.split(".") for token in model.pred_with_transformer)
        if managed and param.dim() == 4 and name.endswith("weight"):
            filters += param.shape[0] * param.shape[1]
    return filters


def _accuracy_by_task(method, test_datasets, task_ids, device):
    return {
        task_id: round(float(method.evaluate(test_datasets[task_id], task_id, device=device)), 4)
        for task_id in task_ids
    }


def handler(args):
    config = vars(args)
    ds_name = config["dataset"]
    print(f"\n##### Baseline: l2l on {ds_name}")

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device

    train_datasets, test_datasets, task_labels, n_tasks = load_split_datasets(config)

    model = base_models.__dict__[config["model_class"]](
        config, n_heads=None, n_classes=config["n"]
    ).to(device)
    train_scope, scope_tokens = apply_pred_strategy(model, config)

    # The base transformer reads config["spatial"]; set it from the managed conv
    # filter count so its filter-wise path lines up exactly.
    filters = _managed_conv_filters(model)
    config["spatial"] = filters if filters > 0 else None

    nweights = l2l_utils.count_parameters(model)
    criterion = nn.CrossEntropyLoss().to(device)
    # Base paper: MNIST datasets use the SVHN-pretrained task encoder; CIFAR
    # uses the ImageNet-pretrained VGG11 encoder, which needs no path.
    te_path = _svhn_task_encoder_path() if "mnist" in ds_name else None
    transformer = l2l_transformer.TransformerModel(nweights, config, te_path).to(device)
    method = L2LMethod(model, transformer, criterion, config)

    base_params = sum(p.numel() for p in model.parameters())
    # transformer_params = sum(p.numel() for p in transformer.parameters())
    transformer_params = sum(
        p.numel()
        for p in transformer.parameters()
        if not isinstance(p, torch.nn.parameter.UninitializedParameter)
    )

    previous_tasks_acc = {}
    future_tasks_acc = {}
    task_times = {}
    task_profiles = {}
    start_time = time.time()

    for task_idx in range(n_tasks):
        task_steps = config["steps"]
        if task_idx == 0:
            task_steps += config["warmup_steps"]
        print(f"Task {task_idx}/{n_tasks} (steps={task_steps})")
        task_start = time.time()

        with Profiler(label=f"task_{task_idx}") as p:
            method.fit(
                train_datasets[task_idx], task_idx,
                save_dir=None, print_output=False,
            )
        task_profiles[task_idx] = p.result.summary()

        previous_tasks_acc[task_idx] = _accuracy_by_task(
            method, test_datasets, range(task_idx + 1), device
        )
        if task_idx + 1 < n_tasks:
            future_tasks_acc[task_idx] = _accuracy_by_task(
                method, test_datasets, range(task_idx + 1, n_tasks), device
            )

        task_times[task_idx] = round(time.time() - task_start, 2)

    total_time = time.time() - start_time
    transformer_params = sum(
        p.numel() for p in transformer.parameters()
    ) # on cifar we needed to do the if not on this to avoid error
    # that made count less by skipping feature extractor
    # this brings it back

    metrics = utils.compute_all_metrics(
        previous_tasks_acc=previous_tasks_acc,
        future_tasks_acc=future_tasks_acc,
        task_times=task_times,
        n_tasks=n_tasks,
        total_time=total_time,
        base_params=base_params,
        transformer_params=transformer_params,
    )

    print(f"avg_acc_final={metrics['average_accuracy_final']}")
    print(f"bwt={metrics['bwt_avg_over_time']}")
    print(f"fwt={metrics['mean_future_accuracy_over_time']}")

    return build_results_row(
        config, "l2l", metrics, train_scope, task_profiles,
        base_params, transformer_params=transformer_params, extra={
            "n_tasks": n_tasks,
            "spatial": config["spatial"],
        }
    )
