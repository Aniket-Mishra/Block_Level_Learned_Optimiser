"""Numerical helpers and learning-rate control for meta-training."""

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.func import functional_call


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


def zeroed_gradients(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


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


def accuracy(pred, y_true):
    return (pred.argmax(1) == y_true).float().mean().item()
