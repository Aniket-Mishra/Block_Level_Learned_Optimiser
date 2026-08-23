import json
import os
import pickle
import random
import time
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import dataset
import numpy as np
import test
import torch
import torch.nn as nn
import utils
from exp_runner_1_1_config import DEFAULT_OUTPUT_DIR
from methods import PROPOSED
from models import base_models, task_encoder_models
from models.base_models import resolve_training_layer_tokens
from models.transformer_models import TransformerModel
from profiler import Profiler

warnings.filterwarnings("ignore", category=UserWarning)


_RESULT_ROW_KEYS = (
    "dataset",
    "method",
    "model_class",
    "seed",
    "training_layers",
    "block_strategy",
    "block_rows",
    "row_scale",
    "alpha",
    "beta",
    "plasticity_scales",
    "use_layer_id",
    "use_weight_stats",
    "use_stability",
    "use_block_signature",
    "use_tensor_embedding",
    "use_signed_gradient_basis",
    "use_ema",
    "ema_objective",
    "use_pos_encoder",
    "ablation_variant",
    "importance_eps",
    "steps",
    "warmup_steps",
    "test_steps",
    "log_interval",
    "checkpoint_interval",
    "embedding_size",
    "n_layers",
    "num_heads",
    "dim_feedforward",
    "dropout",
    "clamping_transformer",
)


def base_dataset_name(config):
    return config.get("base_dataset") or config["dataset"]


def _experiment_label(config):
    """Short, human-readable label, used as a path component."""
    strategy = config.get("block_strategy", "row_wise")
    parts = [strategy[:4]]

    if strategy == "row_wise":
        parts.append(
            f"br{config.get('block_rows')}_rs{config.get('row_scale')}"
        )
    else:
        parts.append(f"a{config.get('alpha')}_b{config.get('beta')}")

    ps = config.get("plasticity_scales")
    parts.append(f"ps{len(ps)}" if ps is not None else "ps0")
    parts.append("lid" if config.get("use_layer_id") else "nolid")
    parts.append("ws" if config.get("use_weight_stats") else "nows")
    parts.append("stab" if config.get("use_stability") else "nostab")
    variant = config.get("ablation_variant")
    if variant:
        parts.append(variant)

    tl_name = config.get("training_layers_name") or _training_layers_label(
        config.get("training_layers")
    )
    parts.append(f"tl-{tl_name}")

    return "_".join(parts)


def _load_data(config, base_name=None):
    if base_name is None:
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
        task_labels = task_labels[:10]

    n_tasks = len(task_labels)
    class_il = bool(config.get("class_il", False))
    train_datasets, test_datasets = dataset.split_task_construction(
        base_name, task_labels, class_il=class_il
    )
    return train_datasets, test_datasets, task_labels, n_tasks


def _apply_pred_strategy(model, config):
    """Sets model.pred_with_transformer from config['training_layers'].

    Returns (train_scope_summary, resolved_tokens). A missing key keeps the
    model's class default.
    """
    if "training_layers" in config:
        tokens = resolve_training_layer_tokens(
            model, config["training_layers"]
        )
        model.pred_with_transformer = tokens
    else:
        tokens = list(model.pred_with_transformer)

    train_scope = utils.summarize_train_scope(
        model,
        include_tokens=tokens,
        min_init_std=config.get("min_init_std", utils.DEFAULT_MIN_INIT_STD),
    )
    return train_scope, tokens


def _build_task_encoder(config, device, base_name=None):
    model_class = config["model_class"]
    if base_name is None:
        base_name = base_dataset_name(config)

    if model_class == "ConvNet":
        if "mnist" in base_name:
            script_dir = Path(__file__).resolve().parent
            mnist_path = (
                script_dir.parent
                / "pretrained_task_encoders"
                / "task_encoder_svhn.pth"
            )
            return task_encoder_models.TaskEncoderMNIST(
                pretrained_model_path=str(mnist_path), config_params=config
            ).to(device)
        return task_encoder_models.TaskEncoderGeneric(config_params=config).to(
            device
        )

    if model_class == "ThreeConvNetSimple":
        return task_encoder_models.TaskEncoderCIFAR(config_params=config).to(
            device
        )

    if model_class == "ResNet18":
        return task_encoder_models.TaskEncoderResNet(config_params=config).to(
            device
        )

    if model_class == "ViTSmall":
        return task_encoder_models.TaskEncoderViT(config_params=config).to(
            device
        )

    return task_encoder_models.TaskEncoderGeneric(config_params=config).to(
        device
    )


def _init_method(config, n_tasks, device, base_name=None):
    print("Initializing the models")
    criterion = nn.CrossEntropyLoss().to(device)

    model_class_name = config["model_class"]
    if not hasattr(base_models, model_class_name):
        raise AttributeError(
            f"Model class '{model_class_name}' not found in models/base_models.py"
        )
    model = getattr(base_models, model_class_name)(
        config, n_heads=None, n_classes=config["n"]
    ).to(device)

    train_scope, tokens = _apply_pred_strategy(model, config)

    print(
        f"selected_total_parameters={train_scope['selected_total_parameters']}\n"
        f"training_scope_label={train_scope['training_scope_label']}\n"
        f"last_trainable_module={train_scope['last_trainable_module']}"
    )
    leaf_names = train_scope["trainable_leaf_module_names"]
    if leaf_names:
        print(
            f"trainable leaf modules ({len(leaf_names)}): "
            + ", ".join(leaf_names)
        )

    block_info = utils.calculate_transformer_blocks(train_scope, config)
    print(
        f"total_blocks={block_info['total_blocks']} "
        f"(strategy={config.get('block_strategy', 'row_wise')}, "
        f"block_rows={config.get('block_rows')}, row_scale={config.get('row_scale')})\n"
        f"max_blocks_in_any_tensor={block_info['max_blocks_in_any_tensor']}"
    )

    task_encoder = _build_task_encoder(config, device, base_name=base_name)
    transformer_model = TransformerModel(
        config_params=config,
        base_model_info=train_scope,
        task_encoder=task_encoder,
    ).to(device)

    method = PROPOSED(model, transformer_model, criterion, config)
    method.base_model_train_scope = train_scope
    method.base_model_block_info = block_info
    method.base_model_pred_with_transformer_tokens = list(tokens)
    return method


def _scalability_metrics(
    block_info, train_scope, base_params, transformer_model
):
    te_params = sum(p.numel() for p in transformer_model.te_model.parameters())
    meta_params = (
        sum(p.numel() for p in transformer_model.parameters()) - te_params
    )
    trans_params = te_params + meta_params

    n_blocks = block_info["total_blocks"]
    n_weights = train_scope["selected_total_parameters"]
    seq_len = n_blocks + 1
    ratio = seq_len / (n_weights + 1)
    return {
        "seq_len": seq_len,
        "seq_len_per_param": n_weights + 1,
        "seq_compression_ratio": round(ratio, 6),
        "attention_complexity_ratio": round(ratio**2, 8),
        "task_encoder_params": te_params,
        "meta_optimizer_params": meta_params,
        "transformer_params": trans_params,
        "meta_overhead_ratio": round(meta_params / base_params, 6)
        if base_params > 0
        else None,
    }


def _profile_summary(task_profiles, seed=0):
    gpu_alloc = [v["gpu_peak_allocated_mb"] for v in task_profiles.values()]
    gpu_res = [v["gpu_peak_reserved_mb"] for v in task_profiles.values()]
    cpu_delta = [v["cpu_delta_mb"] for v in task_profiles.values()]
    time_s = [v["time_s"] for v in task_profiles.values()]

    def stats(vals, prefix):
        return {
            f"{prefix}_mean": round(sum(vals) / len(vals), 2),
            f"{prefix}_max": round(max(vals), 2),
            f"{prefix}_min": round(min(vals), 2),
        }

    series = {
        "gpu_peak_allocated_mb": gpu_alloc,
        "gpu_peak_reserved_mb": gpu_res,
        "cpu_delta_mb": cpu_delta,
        "fit_time_s": time_s,
    }
    summary = {}
    for prefix, vals in series.items():
        summary.update(stats(vals, prefix))
        ci = utils.bootstrap_ci(vals, seed=seed)
        summary[f"{prefix}_mean_ci_low"] = round(ci["ci_low"], 2)
        summary[f"{prefix}_mean_ci_high"] = round(ci["ci_high"], 2)
        summary[f"{prefix}_mean_std_err"] = round(ci["std_err"], 2)
    return summary


def _load_task_scores(scores_path):
    if not scores_path.exists():
        return {}
    try:
        with open(scores_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: could not load existing {scores_path.name}: {e}")
        return {}


def _write_task_scores(scores_path, scores_by_task):
    try:
        with open(scores_path, "wb") as f:
            pickle.dump(scores_by_task, f)
    except Exception as e:
        print(f"Warning: could not save {scores_path.name}: {e}")


def _update_task_scores(scores_path, task_id, scores, scores_by_task):
    scores_by_task[int(task_id)] = {
        name: tensor.detach().cpu() for name, tensor in scores.items()
    }
    _write_task_scores(scores_path, scores_by_task)


def _build_output_paths(base_dir, dataset_name, exp_label, seed, method_name):
    seed_str = f"seed{seed}"
    return {
        "model": base_dir / dataset_name / exp_label / seed_str / "models",
        "task": base_dir
        / dataset_name
        / exp_label
        / seed_str
        / "tasks"
        / method_name,
    }


def _set_reproducibility(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _training_layers_label(training_layers):
    if training_layers is None:
        return "all"
    if isinstance(training_layers, str):
        return training_layers
    return ",".join(training_layers)


def handler(args):
    config = vars(args)
    dataset_name = config["dataset"]
    method_name = config["method"]
    seed = config["seed"]

    print(f"\n##### Running {dataset_name}")

    exp_label = _experiment_label(config)
    print(f"Experiment label: {exp_label}")

    base_output_dir = Path(config.get("output_dir") or DEFAULT_OUTPUT_DIR)
    paths = _build_output_paths(
        base_output_dir, dataset_name, exp_label, seed, method_name
    )
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    config_file = paths["task"] / "config.json"
    if config_file.exists():
        raise RuntimeError(
            f"Run directory already has results: {paths['task']}. "
            "Delete it to re-run this exact config."
        )
    config_file.write_text(json.dumps(config, default=str, indent=2))

    _set_reproducibility(seed)

    device, selected_gpus = utils.set_device()
    config["device"] = device

    base_name = base_dataset_name(config)
    train_datasets, test_datasets, task_labels, n_tasks = _load_data(
        config, base_name=base_name
    )

    label_file = (
        "rotation_angles.json"
        if base_name == "rotatedmnist"
        else "task_labels.json"
    )
    utils.save_object(task_labels, paths["task"] / label_file)

    start_time = time.time()
    method = _init_method(config, n_tasks, device, base_name=base_name)

    previous_tasks_acc, future_tasks_acc, task_times, task_profiles = (
        {},
        {},
        {},
        {},
    )
    converged_steps = {}
    print("Start training")

    # One save_dir per run, not per task: metrics CSVs, task_scores.pkl and
    # checkpoints accumulate at this root, with task_id columns / keys /
    # filenames keeping tasks disambiguated.
    save_dir = paths["task"]
    scores_path = save_dir / "task_scores.pkl"
    is_proposed = method_name == "proposed"
    scores_by_task = _load_task_scores(scores_path) if is_proposed else None
    eval_device = (
        device
        if base_name == "cifar100"
        else (f"cuda:{selected_gpus[1]}" if len(selected_gpus) > 1 else device)
    )

    max_tasks = config.get("max_tasks")
    tasks_to_run = n_tasks if max_tasks is None else min(max_tasks, n_tasks)
    if max_tasks is None:
        print(f"Running all {n_tasks} tasks (max_tasks not set)")
    else:
        print(
            f"Running {tasks_to_run}/{n_tasks} tasks (max_tasks={max_tasks})"
        )

    for task_idx in range(tasks_to_run):
        print(f"Task idx: {task_idx}")
        task_start = time.time()

        with Profiler(label=f"task_{task_idx}") as p:
            method, weights_info = method.fit(
                train_datasets[task_idx], task_idx, save_dir=save_dir
            )
        task_profiles[task_idx] = p.result.summary()
        converged_steps[int(task_idx)] = weights_info.get("converged_step")

        if is_proposed:
            _update_task_scores(
                scores_path, task_idx, weights_info["score"], scores_by_task
            )

        previous_tasks_acc[int(task_idx)] = test.test_previous_tasks(
            method, test_datasets, task_idx, eval_device
        )
        utils.save_object(previous_tasks_acc, save_dir / "test_accuracy.json")

        if task_idx + 1 < n_tasks:
            future_tasks_acc[int(task_idx)] = test.test_future_tasks(
                method, test_datasets, task_idx, n_tasks, eval_device
            )
            utils.save_object(
                future_tasks_acc, save_dir / "future_accuracy.json"
            )

        task_times[task_idx] = round(time.time() - task_start, 2)

    utils.save_object(task_profiles, paths["task"] / "task_profiles.json")
    total_time = time.time() - start_time
    base_params = sum(p.numel() for p in method.model.parameters())
    trans_params = sum(
        p.numel() for p in method.transformer_model.parameters()
    )

    metrics = utils.compute_all_metrics(
        previous_tasks_acc=previous_tasks_acc,
        future_tasks_acc=future_tasks_acc,
        task_times=task_times,
        n_tasks=tasks_to_run,
        total_time=total_time,
        base_params=base_params,
        transformer_params=trans_params,
        bootstrap_seed=config["seed"],
        bootstrap_resamples=config.get("bootstrap_resamples", 1000),
        bootstrap_confidence=config.get("bootstrap_confidence", 0.95),
    )
    print(f"{metrics['average_accuracy_final']=}")
    print(f"Anna's bwt: {metrics['bwt_avg_over_time']=}")
    print(f"Anna's fwt: {metrics['mean_future_accuracy_over_time']=}")

    train_scope = getattr(method, "base_model_train_scope", {})
    block_info = getattr(method, "base_model_block_info", {})

    model_info = {
        "base_model_train_scope": train_scope,
        "base_model_block_info": block_info,
        "base_model_pred_with_transformer_tokens": getattr(
            method, "base_model_pred_with_transformer_tokens", []
        ),
    }

    torch.save(method.model.state_dict(), paths["model"] / "model.pth")
    torch.save(
        method.transformer_model.state_dict(),
        paths["model"] / "transformer_model.pth",
    )
    utils.save_object(model_info, paths["model"] / "model_info.json")

    results_row = {k: config.get(k) for k in _RESULT_ROW_KEYS}
    results_row["training_layers"] = _training_layers_label(
        config.get("training_layers")
    )
    results_row["exp_label"] = exp_label
    results_row["run_dir"] = str(save_dir)
    results_row["n_plasticity_levels"] = (
        len(config["plasticity_scales"])
        if config.get("plasticity_scales")
        else 0
    )
    results_row["total_blocks"] = block_info.get("total_blocks")
    results_row["max_blocks_in_any_tensor"] = block_info.get(
        "max_blocks_in_any_tensor"
    )
    results_row["selected_nweights"] = train_scope.get(
        "selected_total_parameters"
    )
    results_row["selected_nlayers"] = train_scope.get(
        "selected_parameter_tensors"
    )
    results_row["training_scope_label"] = train_scope.get(
        "training_scope_label"
    )
    results_row["base_params"] = base_params
    results_row["transformer_params"] = trans_params
    results_row["converged_step_per_task"] = converged_steps
    utils.save_object(converged_steps, save_dir / "convergence.json")

    results_row.update(
        _scalability_metrics(
            block_info, train_scope, base_params, method.transformer_model
        )
    )
    results_row.update(_profile_summary(task_profiles, seed=config["seed"]))
    results_row.update(metrics)

    return results_row
