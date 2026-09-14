import json
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from scipy import stats
from torch.func import functional_call

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

Batch = namedtuple("Batch", ["x_sp", "y_sp", "x_qr", "y_qr"])


def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), list(range(torch.cuda.device_count()))
    return torch.device("cpu"), []


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def func_call(model, params_dict=None, args=(), kwargs=None):
    """Calls a module, with torch.func.functional_call when params are overridden.

    params_dict=None calls the module directly, skipping the cost of
    rebuilding dict(model.named_parameters()) on every forward.
    """
    if kwargs is None:
        kwargs = {}

    if params_dict is None:
        if isinstance(args, tuple):
            return model(*args, **kwargs)
        return model(args, **kwargs)

    return functional_call(model, params_dict, args, kwargs)


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


def zeroed_gradients(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def define_task_labels(labels, num_classes):
    return [
        labels[i : i + num_classes] for i in range(0, len(labels), num_classes)
    ]


def l2_regularization(dict_parameters):
    # Per-tensor L2 norms, not sum-of-squares: squared L2 produces NaN
    # gradients for zero-initialised tensors.
    return sum(torch.norm(p, p=2) for p in dict_parameters.values())


def select_top_k(importance_scores, K):
    """Keeps the top-K fraction of scores globally and zeros the rest."""
    flat = torch.cat([s.view(-1) for s in importance_scores.values()])
    if flat.numel() == 0:
        return importance_scores.copy()
    threshold = torch.quantile(flat, 1 - K)
    return {
        name: s * (s >= threshold) for name, s in importance_scores.items()
    }


class CustomLRScheduler(lr_scheduler._LRScheduler):
    """Warmup-then-constant outer LR with event-driven decay.

    Decay is triggered by the CoherenceController via decay(). With
    outer_optimizer="prodigy" pass lr_init=1.0 and use_warmup=False so the
    scheduler only applies the decay multiplier.
    """

    def __init__(
        self, optimizer, config_params, task_id, lr_init=None, use_warmup=True
    ):
        self.warmup_steps = config_params["warmup_steps"] if use_warmup else 0
        self.lr_init = (
            lr_init
            if lr_init is not None
            else config_params.get("lr_transformer", config_params["lr"])
        )
        self.task_id = task_id
        self.decay_factor = 1.0
        super().__init__(optimizer)

    def get_lr(self):
        if (
            self.task_id == 0
            and self.warmup_steps > 0
            and self.last_epoch < self.warmup_steps
        ):
            lr = self.lr_init * (self.last_epoch / self.warmup_steps)
        else:
            lr = self.lr_init
        return [lr * self.decay_factor] * len(self.optimizer.param_groups)

    def decay(self, factor=0.5):
        self.decay_factor *= factor
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group["lr"] = lr


class CoherenceController:
    """Stationarity-driven anneal-then-stop controller for the outer loop.

    A diagnostic block is q sign tests, each the inner product of the mean
    meta-gradients of two consecutive `seg`-step sub-windows (Pflug 1990;
    SplitSGD, Sordello & Su 2019). A strict majority of negative signs
    (2*neg > q) declares stationarity: each of the first `max_decays`
    detections halves the outer LR, the next one stops the task. The first
    block of each task is burn-in, since the task-boundary transient mimics
    stationarity. Signs and counts only, so nothing is scale-, architecture-
    or step-budget-dependent.
    """

    DECAY = "decay"
    STOP = "stop"

    def __init__(self, window, q=4, max_decays=3, burn_in=None):
        self.seg = max(1, int(window) // 2)
        self.q = int(q)
        self.max_decays = int(max_decays)
        self.n_decays = 0
        self.burn_in_left = (
            int(burn_in) if burn_in is not None else self.block_len
        )
        self._signs = []
        self._step_in_pair = 0
        self._sum_first = None
        self._sum_second = None
        self.last_coherence = None
        self.last_neg_votes = None
        self.steps_seen = 0

    @property
    def block_len(self):
        return self.q * 2 * self.seg

    def _reset_pair(self):
        self._step_in_pair = 0
        self._sum_first = None
        self._sum_second = None

    def update(self, grad_vec):
        """Feeds one flattened meta-gradient; returns None, DECAY, or STOP."""
        self.steps_seen += 1
        if self.burn_in_left > 0:
            self.burn_in_left -= 1
            return None

        g = grad_vec.detach()
        if self._step_in_pair < self.seg:
            self._sum_first = (
                g.clone() if self._sum_first is None else self._sum_first + g
            )
        else:
            self._sum_second = (
                g.clone() if self._sum_second is None else self._sum_second + g
            )
        self._step_in_pair += 1
        if self._step_in_pair < 2 * self.seg:
            return None

        coherence = torch.dot(
            self._sum_first / self.seg, self._sum_second / self.seg
        ).item()
        self.last_coherence = coherence
        self._signs.append(coherence < 0)
        self._reset_pair()
        if len(self._signs) < self.q:
            return None

        neg_votes = sum(self._signs)
        self._signs = []
        self.last_neg_votes = neg_votes
        if 2 * neg_votes <= self.q:
            return None
        if self.n_decays < self.max_decays:
            self.n_decays += 1
            return self.DECAY
        return self.STOP


# Continual-learning metrics. Accuracy matrices are dicts keyed by training
# time T mapping to {task_id: accuracy}, written R_{T, i} below.


def accuracy(pred, y_true):
    return (pred.argmax(1) == y_true).float().mean().item()


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
