import json
from collections import namedtuple

import numpy as np
import torch
from scipy import stats

from block_level_learned_optimization.task_encoder import LambdaLayer as LambdaLayer
from block_level_learned_optimization.optimizer_setup import (
    make_transformer_optimizer as make_transformer_optimizer,
    set_optimizer as set_optimizer,
)
from block_level_learned_optimization.parameter_scope import (
    DEFAULT_MIN_INIT_STD as DEFAULT_MIN_INIT_STD,
    _is_leaf_module as _is_leaf_module,
    _matches_filters as _matches_filters,
    _name_matches_any_token as _name_matches_any_token,
    _segments_contain as _segments_contain,
    calculate_transformer_blocks as calculate_transformer_blocks,
    summarize_train_scope as summarize_train_scope,
)
from block_level_learned_optimization.training_utils import (
    CoherenceController as CoherenceController,
    CustomLRScheduler as CustomLRScheduler,
    accuracy as accuracy,
    func_call as func_call,
    l2_regularization as l2_regularization,
    select_top_k as select_top_k,
    zeroed_gradients as zeroed_gradients,
)

Batch = namedtuple("Batch", ["x_sp", "y_sp", "x_qr", "y_qr"])


def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), list(range(torch.cuda.device_count()))
    return torch.device("cpu"), []


def save_object(obj, name):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                obj[k] = v.tolist()
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, np.ndarray):
                        obj[k][k2] = v2.tolist()
    with open(name, "w") as f:
        json.dump(obj, f)


def load_object(name):
    with open(name, "r") as f:
        return json.load(f)


def define_task_labels(labels, num_classes):
    return [
        labels[i : i + num_classes] for i in range(0, len(labels), num_classes)
    ]


# Continual-learning metrics. Accuracy matrices are dicts keyed by training
# time T mapping to {task_id: accuracy}, written R_{T, i} below.


def infer_n_tasks_from_previous(previous_tasks_acc):
    int_keys = [k for k in previous_tasks_acc if isinstance(k, int)]
    return (max(int_keys) + 1) if int_keys else 0


def compute_final_acc_per_task(previous_tasks_acc, n_tasks):
    """R_{T-1, i}: accuracy on task i after training all tasks."""
    final_row = previous_tasks_acc.get(n_tasks - 1, {})
    return {i: float(final_row[i]) for i in range(n_tasks) if i in final_row}


def compute_peak_acc_per_task(previous_tasks_acc, n_tasks):
    """R_{i, i}: accuracy on task i immediately after learning it."""
    return {
        i: float(previous_tasks_acc[i][i])
        for i in range(n_tasks)
        if i in previous_tasks_acc and i in previous_tasks_acc[i]
    }


def compute_forward_accuracy_per_task(future_tasks_acc, n_tasks):
    """R_{i-1, i} for i > 0: accuracy on task i before training on it."""
    out = {0: 0.0}
    for i in range(1, n_tasks):
        row = future_tasks_acc.get(i - 1, {})
        if i in row:
            out[i] = float(row[i])
    return out


def compute_average_accuracy(previous_tasks_acc, n_tasks):
    """Mean of R_{T-1, i} over all tasks. Standard CL average accuracy."""
    vals = list(
        compute_final_acc_per_task(previous_tasks_acc, n_tasks).values()
    )
    return float(sum(vals) / len(vals)) if vals else 0.0


def compute_learning_accuracy_mean(previous_tasks_acc, n_tasks):
    """Mean of R_{i, i}: how well the model learns each task when first trained on it."""
    vals = list(
        compute_peak_acc_per_task(previous_tasks_acc, n_tasks).values()
    )
    return float(sum(vals) / len(vals)) if vals else 0.0


def compute_bwt_per_task(previous_tasks_acc, n_tasks):
    """BWT per task: R_{T-1, i} - R_{i, i}. Negative means forgetting."""
    final = compute_final_acc_per_task(previous_tasks_acc, n_tasks)
    peak = compute_peak_acc_per_task(previous_tasks_acc, n_tasks)
    return {
        i: float(final[i] - peak[i])
        for i in range(n_tasks)
        if i in final and i in peak
    }


def compute_BWT_final_mean(previous_tasks_acc, n_tasks):
    """Mean BWT across tasks. Negative = net forgetting, positive = net improvement."""
    vals = list(compute_bwt_per_task(previous_tasks_acc, n_tasks).values())
    return float(sum(vals) / len(vals)) if vals else 0.0


def compute_forward_accuracy_mean(future_tasks_acc, n_tasks):
    """Mean forward accuracy over tasks i=1..T-1."""
    fa = compute_forward_accuracy_per_task(future_tasks_acc, n_tasks)
    vals = [v for k, v in fa.items() if isinstance(k, int) and k != 0]
    return float(sum(vals) / len(vals)) if vals else 0.0


def compute_forgetting(previous_tasks_acc, n_tasks):
    """F_i = max_{t >= i} R_{t, i} - R_{T-1, i}. Returns (mean, per-task dict)."""
    final_row = previous_tasks_acc.get(n_tasks - 1, {})
    forgetting_per_task = {}
    for i in range(n_tasks):
        if i not in final_row:
            continue
        seen = [
            float(previous_tasks_acc[t][i])
            for t in range(i, n_tasks)
            if t in previous_tasks_acc and i in previous_tasks_acc[t]
        ]
        if seen:
            forgetting_per_task[i] = float(max(seen) - float(final_row[i]))

    vals = list(forgetting_per_task.values())
    return (float(sum(vals) / len(vals)) if vals else 0.0), forgetting_per_task


def compute_bwt(tasks_accuracies):
    """Mean of R_{T, i} - R_{i, i} over all (task, time) pairs with T > i."""
    total, n = 0.0, 0.0
    for T in tasks_accuracies:
        for i in range(T):
            if i in tasks_accuracies[T] and i in tasks_accuracies.get(i, {}):
                total += tasks_accuracies[T][i] - tasks_accuracies[i][i]
                n += 1
    return total / n if n > 0 else 0.0


def compute_fwt(future_tasks_acc):
    """Mean accuracy on unseen tasks over all future (task, time) pairs."""
    total, n = 0.0, 0.0
    n_tasks = infer_n_tasks_from_previous(future_tasks_acc)
    for T in future_tasks_acc:
        for i in range(T, n_tasks):
            if i in future_tasks_acc[T]:
                total += future_tasks_acc[T][i]
                n += 1
    return total / n if n > 0 else 0.0


def bootstrap_ci(values, n_resamples=1000, confidence_level=0.95, seed=0):
    """Percentile bootstrap CI for the mean of a 1D sample.

    Reflects spread across the resampled units, not seed-to-seed variation.
    Uses an isolated, seeded generator so the global training RNG is never
    perturbed.
    """
    clean = [float(v) for v in values if v is not None]
    if len(clean) < 2 or min(clean) == max(clean):
        point = clean[0] if clean else 0.0
        return {
            "mean": point,
            "std_err": 0.0,
            "ci_low": point,
            "ci_high": point,
        }

    result = stats.bootstrap(
        (clean,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
        random_state=seed,
    )
    return {
        "mean": float(np.mean(clean)),
        "std_err": float(result.standard_error),
        "ci_low": float(result.confidence_interval.low),
        "ci_high": float(result.confidence_interval.high),
    }


def compute_all_metrics(
    previous_tasks_acc,
    future_tasks_acc,
    task_times=None,
    *,
    n_tasks=None,
    total_time=None,
    base_params=None,
    transformer_params=None,
    bootstrap_seed=0,
    bootstrap_resamples=1000,
    bootstrap_confidence=0.95,
):
    """Computes all CL metrics in a single flat dict suitable for CSV export."""
    if n_tasks is None:
        n_tasks = infer_n_tasks_from_previous(previous_tasks_acc)

    final_acc_per_task = compute_final_acc_per_task(
        previous_tasks_acc, n_tasks
    )
    peak_acc_per_task = compute_peak_acc_per_task(previous_tasks_acc, n_tasks)
    bwt_per_task = compute_bwt_per_task(previous_tasks_acc, n_tasks)
    fa_per_task = compute_forward_accuracy_per_task(future_tasks_acc, n_tasks)
    forgetting_mean, forgetting_per_task = compute_forgetting(
        previous_tasks_acc, n_tasks
    )
    time_per_task = task_times or {}
    times = [float(v) for v in time_per_task.values()]

    bwt_over_time_sample = [
        previous_tasks_acc[T][i] - previous_tasks_acc[i][i]
        for T in previous_tasks_acc
        for i in range(T)
        if i in previous_tasks_acc[T] and i in previous_tasks_acc.get(i, {})
    ]
    fwt_over_time_sample = [
        future_tasks_acc[T][i]
        for T in future_tasks_acc
        for i in range(T, n_tasks)
        if i in future_tasks_acc[T]
    ]

    final_vals = list(final_acc_per_task.values())
    peak_vals = list(peak_acc_per_task.values())
    bwt_vals = list(bwt_per_task.values())
    fa_vals_no_task0 = [v for k, v in fa_per_task.items() if k != 0]
    forgetting_vals = list(forgetting_per_task.values())

    ci_samples = {
        "average_accuracy_final": final_vals,
        "learning_accuracy_mean": peak_vals,
        "BWT_final_mean": bwt_vals,
        "forward_accuracy_mean": fa_vals_no_task0,
        "forgetting_mean": forgetting_vals,
        "retention_mean": [1.0 - v for v in forgetting_vals],
        "bwt_avg_over_time": bwt_over_time_sample,
        "mean_future_accuracy_over_time": fwt_over_time_sample,
        "mean_task_time": times,
    }
    ci_columns = {}
    for name, sample in ci_samples.items():
        ci = bootstrap_ci(
            sample, bootstrap_resamples, bootstrap_confidence, bootstrap_seed
        )
        ci_columns[f"{name}_ci_low"] = round(ci["ci_low"], 4)
        ci_columns[f"{name}_ci_high"] = round(ci["ci_high"], 4)
        ci_columns[f"{name}_std_err"] = round(ci["std_err"], 4)

    average_accuracy_final = compute_average_accuracy(
        previous_tasks_acc, n_tasks
    )
    learning_accuracy_mean = compute_learning_accuracy_mean(
        previous_tasks_acc, n_tasks
    )
    bwt_final_mean = compute_BWT_final_mean(previous_tasks_acc, n_tasks)
    forward_accuracy_mean = compute_forward_accuracy_mean(
        future_tasks_acc, n_tasks
    )

    return {
        "test_accuracy": previous_tasks_acc,
        "future_tasks_acc": future_tasks_acc,
        "final_acc_per_task": final_acc_per_task,
        "peak_acc_per_task": peak_acc_per_task,
        "bwt_per_task": bwt_per_task,
        "forward_accuracy_per_task": fa_per_task,
        "forgetting_per_task": forgetting_per_task,
        "time_per_task": time_per_task,
        "average_accuracy_final": round(average_accuracy_final, 4),
        "learning_accuracy_mean": round(learning_accuracy_mean, 4),
        "BWT_final_mean": round(bwt_final_mean, 4),
        "forward_accuracy_mean": round(forward_accuracy_mean, 4),
        "forgetting_mean": round(forgetting_mean, 4),
        "retention_mean": round(1.0 - forgetting_mean, 4),
        "bwt_avg_over_time": round(compute_bwt(previous_tasks_acc), 4),
        "mean_future_accuracy_over_time": round(
            compute_fwt(future_tasks_acc), 4
        ),
        "total_time": round(float(total_time), 2)
        if total_time is not None
        else None,
        "base_params": int(base_params) if base_params is not None else None,
        "transformer_params": int(transformer_params)
        if transformer_params is not None
        else None,
        "mean_task_time": round(sum(times) / len(times), 2) if times else 0.0,
        "min_task_time": round(min(times), 2) if times else 0.0,
        "max_task_time": round(max(times), 2) if times else 0.0,
        **ci_columns,
    }
