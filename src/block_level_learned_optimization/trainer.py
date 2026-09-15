"""The proposed transformer meta-optimizer for continual learning."""

import copy
import csv
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .optimizer_setup import (
    make_transformer_optimizer,
    set_optimizer,
)
from .parameter_scope import (
    DEFAULT_MIN_INIT_STD,
    _name_matches_any_token,
)
from .training_utils import (
    CoherenceController,
    CustomLRScheduler,
    accuracy,
    func_call,
    l2_regularization,
    select_top_k,
    zeroed_gradients,
)

_METRICS_STEPS_FIELDS = (
    "task_id",
    "step",
    "loss",
    "inner_loss",
    "dummy_loss",
    "l2reg",
    "lr",
    "c",
    "meta_grad_cosine_sim",
    "meta_grad_dead_frac_mean",
    "meta_grad_norm",
    "weight_update_layer_mean",
    "ms_per_step_c",
    "ms_per_step_inner",
    "ms_per_step_outer_fwd",
    "ms_per_step_outer_bwd",
    "ms_per_step_total",
    "lr_decay_factor",
    "ctrl_n_decays",
    "ctrl_last_coherence",
    "ctrl_last_neg_votes",
)

# One row per decay/stop event plus exactly one terminal row per task
# (event = "converged" or "budget_cap"). The terminal row's `step` is the
# per-task step budget actually consumed; read that row when assigning step
# budgets to the baselines.
_CONTROLLER_FIELDS = (
    "task_id",
    "step",
    "event",
    "coherence",
    "neg_votes",
    "q",
    "n_decays",
    "lr_decay_factor",
    "controller_window",
    "max_decays",
    "budget_cap",
    "wall_time_s",
)

_METRICS_PER_TENSOR_FIELDS = (
    "task_id",
    "step",
    "tensor_name",
    "upd_abs_mean",
    "applied_abs_mean",
    "clamp_frac",
)


def estimate_importance(c, model, managed_names):
    weights_info = {"data": {}, "grad": {}, "score": {}}
    min_importance = float("inf")
    max_importance = float("-inf")
    for name, param in model.named_parameters():
        if name not in managed_names:
            param.grad = None
            continue

        if not param.requires_grad or param.grad is None:
            continue

        grad = param.grad.clone()
        weights_info["data"][name] = param.data.clone()
        weights_info["grad"][name] = grad
        if isinstance(c, dict):
            importance = grad.abs() * c[name]
        else:
            importance = grad.abs()
        weights_info["score"][name] = importance
        min_importance = min(min_importance, torch.min(importance))
        max_importance = max(max_importance, torch.max(importance))
        param.grad = None

    denom = max(
        max_importance - min_importance,
        torch.tensor(
            1e-12,
            device=next(iter(weights_info["score"].values())).device,
        ),
    )
    for name, importance in weights_info["score"].items():
        weights_info["score"][name] = (importance - min_importance) / denom
    return weights_info


def _append_csv_rows(path, fields, rows):
    if path is None or not rows:
        return
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _controller_row(
    task, step, event, controller, scheduler, budget_cap, fit_start_time
):
    return {
        "task_id": int(task),
        "step": step,
        "event": event,
        "coherence": (
            controller.last_coherence
            if controller.last_coherence is not None
            else ""
        ),
        "neg_votes": (
            controller.last_neg_votes
            if controller.last_neg_votes is not None
            else ""
        ),
        "q": controller.q,
        "n_decays": controller.n_decays,
        "lr_decay_factor": scheduler.decay_factor,
        "controller_window": 2 * controller.seg,
        "max_decays": controller.max_decays,
        "budget_cap": budget_cap,
        "wall_time_s": time.perf_counter() - fit_start_time,
    }


class PROPOSED:
    def __init__(
        self, model, transformer, criterion, config_params, batch_generator_class
    ):
        self.model = model
        self.transformer_model = transformer
        self.criterion = criterion
        self.config_params = config_params
        self._batch_generator_class = batch_generator_class

        # Hot-path caches; treated as read-only after construction.
        self._mse_loss = nn.MSELoss()
        self._lr_inner = self.config_params["lr_inner"]
        self._lambda_l2 = self.config_params["lambda_l2"]
        self._top_k = self.config_params["top_K"]
        self._inner_steps = self.config_params["inner_steps"]
        self._test_steps = self.config_params["test_steps"]
        self._device = self.config_params["device"]

        self.model_optimizer, self.transformer_optimizer = set_optimizer(
            self.model, self.transformer_model, config_params
        )
        self._outer_opt_name = str(
            config_params.get("outer_optimizer", "adam")
        ).lower()

        # Managed-tensor cache. The init_std floor drops constant-init tensors
        # (BatchNorm weight/bias, biases). It is read from config so this
        # method and summarize_train_scope share the single knob; if they
        # disagree, the transformer output length will not match the number of
        # updated weights (see update_model_parameters).
        token_matched_names = {
            name
            for name, _ in self.model.named_parameters()
            if _name_matches_any_token(
                name, self.model.pred_with_transformer
            )
        }
        self._init_std = {}
        self._weight_clamp = {}
        managed_names = []
        min_init_std = config_params.get(
            "min_init_std", DEFAULT_MIN_INIT_STD
        )
        for name, param in self.model.named_parameters():
            if name not in token_matched_names:
                continue
            init_std = param.data.std().item()
            if init_std < min_init_std:
                continue
            managed_names.append(name)
            self._init_std[name] = init_std
            # Clamp in init_std units, no absolute floor.
            self._weight_clamp[name] = 30 * init_std
        self._transformer_managed_names = frozenset(managed_names)
        self.theta = {
            "model": {
                n: w
                for n, w in self.model.named_parameters()
                if n not in self._transformer_managed_names
            },
            "transformer": {
                n: w
                for n, w in self.transformer_model.named_parameters()
                if w.requires_grad
            },
        }

        self.previous_tsk_score = None
        self.task = 0

        self._use_ema = bool(config_params.get("use_ema", False))
        self._ema_beta = float(config_params.get("ema_beta", 0.95))
        self._ema_objective = str(
            config_params.get("ema_objective", "mas")
        ).lower()
        if self._use_ema and self._ema_objective not in ("ce", "mas"):
            raise ValueError(
                f"ema_objective must be 'ce' or 'mas', got {self._ema_objective!r}"
            )

        # Per-parameter EMA of |grad|, materialised only when the flag is on.
        self._ema = {}
        if self._use_ema:
            for name, param in self.model.named_parameters():
                if name in self._transformer_managed_names:
                    self._ema[name] = torch.zeros_like(param.data)

        # Perf caches, built once for self.model and bypassed when the caller
        # passes a different model (adapt_and_evaluate's copy). Parameter
        # objects update in place across optimizer steps, so caching is safe.
        self._main_model_named_params = list(self.model.named_parameters())
        self._main_model_param_lookup = dict(self._main_model_named_params)
        self._main_model_managed_params = [
            (name, param)
            for name, param in self._main_model_named_params
            if name in self._transformer_managed_names
        ]
        self._main_model_parameters = [
            param for _, param in self._main_model_named_params
        ]

        self._main_transformer_named_params = list(
            self.transformer_model.named_parameters()
        )
        self._main_transformer_param_lookup = dict(
            self._main_transformer_named_params
        )
        self._main_transformer_parameters = [
            param for _, param in self._main_transformer_named_params
        ]

        # With the EMA on, each task's normalised EMA is summed into a frozen
        # running importance at the task boundary (online EWC). prev_scores
        # reads this sum, not the live EMA, so protection reflects every past
        # task rather than the last few steps.
        self._consolidated = {}
        self._consolidated_norm = None
        if self._use_ema:
            for name, param in self.model.named_parameters():
                if name in self._transformer_managed_names:
                    self._consolidated[name] = torch.zeros_like(param.data)

        # Previous outer step's flattened transformer gradient, for the
        # step-to-step cosine similarity diagnostic. Reset per task in fit().
        self._prev_meta_grad = None

    def _managed_params_for(self, model):
        if model is self.model:
            return self._main_model_managed_params
        return [
            (name, param)
            for name, param in model.named_parameters()
            if name in self._transformer_managed_names
        ]

    def _update_ema(self, batch, theta, model):
        """One EMA step on |grad| per managed param.

        Objective "ce" is the supervised loss gradient; "mas" is the gradient
        of the sum-of-squares output norm (label-free). torch.autograd.grad
        leaves param.grad untouched so the importance pass is not clobbered.
        """
        if model is self.model:
            model_params = self._main_model_param_lookup.copy()
        else:
            model_params = dict(model.named_parameters())
        model_params.update(theta["model"])
        pred = func_call(model, model_params, batch.x_sp)

        if self._ema_objective == "ce":
            objective = self.criterion(pred, batch.y_sp)
        else:
            objective = pred.pow(2).sum(dim=tuple(range(1, pred.dim()))).mean()

        targets = []
        target_names = []
        for name, param in self._managed_params_for(model):
            if param.requires_grad:
                targets.append(param)
                target_names.append(name)
        if not targets:
            return
        grads = torch.autograd.grad(
            objective, targets, retain_graph=False, allow_unused=True
        )
        beta = self._ema_beta
        with torch.no_grad():
            for name, g in zip(target_names, grads):
                if g is None:
                    continue
                self._ema[name].mul_(beta).add_(g.abs(), alpha=1.0 - beta)

    def _minmax(self, scores):
        """Min-max a per-parameter dict into [0, 1]."""
        if not scores:
            return None
        flat = torch.cat([s.flatten() for s in scores.values()])
        lo, hi = flat.min(), flat.max()
        denom = (hi - lo).clamp(min=1e-12)
        return {n: (s - lo) / denom for n, s in scores.items()}

    def compute_c(self, step, tot_steps):
        # c=None means all-ones, which estimate_importance treats as the
        # identity, so no ones tensors are allocated. Protection depends on
        # accumulated importance only, never the step count:
        # c_r = 1 - (1 - c0) * s_r with s_r in [0, 1]. Weights unimportant to
        # past tasks keep full plasticity (c = 1); the most important are
        # damped to the floor c0. This is per-parameter soft gradient masking
        # (SPG, Konishi et al., ICML 2023) with importance-proportional
        # strength as in EWC/SI/MAS.
        if self.task == 0:
            return None, 1.0
        if self._use_ema:
            if self._consolidated_norm is None:
                self._consolidated_norm = self._minmax(self._consolidated)
            prev = self._consolidated_norm
        else:
            prev = self.previous_tsk_score
        if prev is None:
            return None, 1.0

        c0 = self.config_params["c"]
        c = {
            name: 1.0 - (1.0 - c0) * prev[name]
            for name, _ in self._main_model_managed_params
        }
        return c, c0

    def compute_importance_score(self, theta, batch, c, model):
        if model is self.model:
            model_params = self._main_model_param_lookup.copy()
        else:
            model_params = dict(model.named_parameters())
        model_params.update(theta["model"])

        zeroed_gradients(model)

        pred = func_call(model, model_params, batch.x_sp)

        loss = self.criterion(pred, batch.y_sp)
        loss.backward()

        weights_info = estimate_importance(
            c, model, managed_names=self._transformer_managed_names
        )

        if self._top_k != 1.0 and self.task > 0:
            weights_info["score"] = select_top_k(
                weights_info["score"], self._top_k
            )

        return weights_info

    def update_model_parameters(self, weight_updates, model):
        update_counter = 0
        updated_parameters = {}
        for name, param in self._managed_params_for(model):
            param_data = param.data
            param_size = param.numel()
            try:
                layer_update = weight_updates[
                    update_counter : update_counter + param_size
                ].reshape(param.shape)
            except RuntimeError as e:
                print(f"Shape mismatch on {name}: {e}")
                layer_update = torch.zeros_like(param)

            # Scale by this tensor's init_std so a transformer output of
            # magnitude 1.0 means "one init standard deviation" in every layer.
            layer_update = layer_update * self._init_std[name]
            param_data += layer_update
            clamp_limit = self._weight_clamp[name]
            param_data.clamp_(-clamp_limit, clamp_limit)
            updated_parameters[name] = param_data.clone()

            update_counter += param_size
        assert update_counter == len(weight_updates), (
            "Different number of weights and weight updates"
        )
        return updated_parameters

    def get_updated_params(
        self,
        params_dict,
        batch,
        c,
        model=None,
        transformer_model=None,
        device=None,
    ):
        if model is None:
            model = self.model
        if transformer_model is None:
            transformer_model = self.transformer_model

        weights_info = self.compute_importance_score(
            params_dict, batch, c, model
        )

        if transformer_model is self.transformer_model:
            transformer_params = self._main_transformer_param_lookup.copy()
        else:
            transformer_params = dict(transformer_model.named_parameters())
        transformer_params.update(params_dict["transformer"])

        # With the EMA on, the consolidated past-task importance is routed
        # through the same prev_importance_scores slot the stability head
        # reads; the wire format is unchanged. Cached because it only changes
        # at a task boundary.
        if self._use_ema and self.task > 0:
            if self._consolidated_norm is None:
                self._consolidated_norm = self._minmax(self._consolidated)
            prev_scores = self._consolidated_norm
        else:
            prev_scores = self.previous_tsk_score if self.task > 0 else None

        output_task, weight_updates = func_call(
            transformer_model,
            transformer_params,
            args=(batch.x_sp, weights_info["data"], weights_info["score"]),
            kwargs={
                "src_mask": None,
                "device": device,
                "prev_importance_scores": prev_scores,
                "raw_gradients": weights_info["grad"],
                "init_std_map": self._init_std,
            },
        )
        del transformer_params
        weights_info["update"] = weight_updates

        updated_model_params = self.update_model_parameters(
            weight_updates, model
        )
        updated_model_params.update(params_dict["model"])
        return output_task, updated_model_params, weights_info

    def one_inner_loop(self, params_dict, batch, task_id, c, device=None):
        output_task, updated_model_params, weights_info = (
            self.get_updated_params(
                params_dict,
                batch,
                c,
                self.model,
                self.transformer_model,
                device,
            )
        )
        predictions = func_call(
            self.model, updated_model_params, batch.x_sp
        )
        inner_loss = self.criterion(predictions, batch.y_sp)
        # dummy_loss exists only to attach gradients to the transformer.
        dummy_loss = self._mse_loss(output_task.squeeze(), task_id)
        l2reg = l2_regularization(updated_model_params)
        inner_loss = inner_loss + 1e-3 * dummy_loss + self._lambda_l2 * l2reg

        all_params_dict = {
            **params_dict["model"],
            **params_dict["transformer"],
        }
        grads = torch.autograd.grad(inner_loss, all_params_dict.values())

        new_params_dict = {"model": {}, "transformer": {}}
        for (name, w), w_grad in zip(all_params_dict.items(), grads):
            if name not in updated_model_params:
                new_params_dict["transformer"][name] = (
                    w - self._lr_inner * w_grad
                )
                continue

            if name in self._transformer_managed_names:
                continue

            new_params_dict["model"][name] = w - self._lr_inner * w_grad
        del updated_model_params
        return new_params_dict, weights_info, inner_loss

    def inner_loop(self, batch, task_id, c, device=None):
        new_theta, _, _ = self.one_inner_loop(
            self.theta, batch, task_id, c, device=device
        )
        for _ in range(self._inner_steps - 2):
            new_theta, _, _ = self.one_inner_loop(
                new_theta, batch, task_id, c, device=device
            )
        return new_theta

    def _compute_meta_grad_diagnostics(self):
        """Returns (cosine_sim, dead_frac_mean, grad_norm) for the transformer
        gradient, read after loss.backward() and before clipping. cosine_sim
        compares against the previous step (None on a task's first step); a
        healthy meta-signal stays positive and stable. The dead fraction is
        the share of gradient entries below 1e-7, averaged over tensors.
        """
        grad_parts = []
        dead_fracs = []
        for p in self._main_transformer_parameters:
            if p.grad is None:
                continue
            g = p.grad.detach()
            grad_parts.append(g.flatten())
            dead_fracs.append((g.abs() < 1e-7).float().mean().item())

        if not grad_parts:
            return None, 0.0, 0.0

        grad_vec = torch.cat(grad_parts)
        dead_frac_mean = float(np.mean(dead_fracs)) if dead_fracs else 0.0
        grad_norm = grad_vec.norm().item()

        cosine_sim = None
        if (
            self._prev_meta_grad is not None
            and self._prev_meta_grad.numel() == grad_vec.numel()
        ):
            cosine_sim = F.cosine_similarity(
                grad_vec.unsqueeze(0), self._prev_meta_grad.unsqueeze(0)
            ).item()
        self._prev_meta_grad = grad_vec.clone()

        return cosine_sim, dead_frac_mean, grad_norm

    def _build_log_records(
        self,
        step,
        loss,
        inner_loss,
        dummy_loss,
        l2reg,
        c_value,
        weights_info,
        timer_accum,
        window_steps,
        meta_grad_cosine_sim,
        meta_grad_dead_frac_mean,
        meta_grad_norm,
    ):
        """Snapshots scalars and per-tensor stats into CSV-ready dict rows."""
        current_lr = self.transformer_optimizer.param_groups[0]["lr"]
        timer_ms = {k: 1000 * v / window_steps for k, v in timer_accum.items()}
        timer_ms["total"] = sum(timer_ms.values())

        step_row = {
            "task_id": int(self.task),
            "step": step + 1,
            "loss": loss.item(),
            "inner_loss": inner_loss.item(),
            "dummy_loss": dummy_loss.item(),
            "l2reg": l2reg.item(),
            "lr": current_lr,
            "c": c_value,
            "meta_grad_cosine_sim": (
                meta_grad_cosine_sim
                if meta_grad_cosine_sim is not None
                else ""
            ),
            "meta_grad_dead_frac_mean": meta_grad_dead_frac_mean,
            "meta_grad_norm": meta_grad_norm,
            "ms_per_step_c": timer_ms["c"],
            "ms_per_step_inner": timer_ms["inner"],
            "ms_per_step_outer_fwd": timer_ms["outer_fwd"],
            "ms_per_step_outer_bwd": timer_ms["outer_bwd"],
            "ms_per_step_total": timer_ms["total"],
        }

        per_tensor_rows = []
        cache = self.transformer_model._block_cache
        if cache is not None and "update" in weights_info:
            update_flat = weights_info["update"].detach()
            model_params = self._main_model_param_lookup
            for name, numel in zip(cache["names"], cache["tensor_numels"]):
                offset = cache["tensor_offsets"][name]
                update = update_flat[offset : offset + numel]
                upd_abs_mean = update.abs().mean().item()
                applied_abs_mean = upd_abs_mean * self._init_std.get(name, 0.0)
                clamp_limit = self._weight_clamp.get(name)
                if clamp_limit is not None and name in model_params:
                    w = model_params[name].data
                    clamp_frac = (
                        (w.abs() >= clamp_limit - 1e-8).float().mean().item()
                    )
                else:
                    clamp_frac = 0.0
                per_tensor_rows.append(
                    {
                        "task_id": int(self.task),
                        "step": step + 1,
                        "tensor_name": name,
                        "upd_abs_mean": upd_abs_mean,
                        "applied_abs_mean": applied_abs_mean,
                        "clamp_frac": clamp_frac,
                    }
                )

        if per_tensor_rows:
            step_row["weight_update_layer_mean"] = sum(
                row["applied_abs_mean"] for row in per_tensor_rows
            ) / len(per_tensor_rows)
        else:
            step_row["weight_update_layer_mean"] = ""

        return step_row, per_tensor_rows, timer_ms

    def _print_log(self, step_row, per_tensor_rows, timer_ms, window_steps):
        cos = step_row["meta_grad_cosine_sim"]
        cos_str = f"{cos:.4f}" if isinstance(cos, float) else "n/a"
        print(
            "Train Step: {}. Loss: {:.4f}, Inner_loss: {:.4f}, "
            "Dummy loss: {:.4f}, l2reg: {:.4f}, lr: {:.6f}, c: {:.1f}, "
            "meta_cos: {}, meta_dead: {:.4f}".format(
                step_row["step"],
                step_row["loss"],
                step_row["inner_loss"],
                step_row["dummy_loss"],
                step_row["l2reg"],
                step_row["lr"],
                step_row["c"],
                cos_str,
                step_row["meta_grad_dead_frac_mean"],
            )
        )
        if per_tensor_rows:
            parts = [
                f"{r['tensor_name']} |upd|={r['upd_abs_mean']:.5f}, "
                f"|applied|={r['applied_abs_mean']:.5f} "
                f"clamp={100 * r['clamp_frac']:.1f}%"
                for r in per_tensor_rows
            ]
            print("  per-tensor: " + " | ".join(parts))
        print(
            "  timer (last {} steps, avg ms/step): "
            "c+batch={:.2f}, inner={:.2f}, outer_fwd={:.2f}, "
            "outer_bwd={:.2f}, total={:.2f}".format(
                window_steps,
                timer_ms["c"],
                timer_ms["inner"],
                timer_ms["outer_fwd"],
                timer_ms["outer_bwd"],
                timer_ms["total"],
            )
        )

    def fit(self, data, task_id, save_dir=None, print_output=True):
        log_interval = int(self.config_params.get("log_interval", 500))
        checkpoint_interval = int(
            self.config_params.get("checkpoint_interval", 5000)
        )

        self.task = int(task_id)
        task_id_tensor = torch.tensor(
            self.task,
            dtype=torch.float,
            device=self._device,
        )

        # Reset so the first logged cosine similarity does not compare across a task boundary.
        self._prev_meta_grad = None

        use_prodigy = self._outer_opt_name == "prodigy"
        if use_prodigy and self.task > 0:
            # this is not used, I got bored of the paper, we skip.
            self.transformer_optimizer = make_transformer_optimizer(
                self.transformer_model, self.config_params
            )

        transformer_scheduler = CustomLRScheduler(
            self.transformer_optimizer,
            self.config_params,
            self.task,
            lr_init=1.0 if use_prodigy else None,
            use_warmup=not use_prodigy,
        )
        current_task_steps = (
            self.config_params["warmup_steps"] + self.config_params["steps"]
            if self.task == 0
            else self.config_params["steps"]
        )

        batch_generator = self._batch_generator_class(data, self.config_params)

        timer_accum = {
            "c": 0.0,
            "inner": 0.0,
            "outer_fwd": 0.0,
            "outer_bwd": 0.0,
        }
        timer_window_start = 0

        metrics_steps_path = (
            save_dir / "metrics_steps.csv" if save_dir is not None else None
        )
        metrics_per_tensor_path = (
            save_dir / "metrics_per_tensor.csv"
            if save_dir is not None
            else None
        )

        buffered_step_rows = []
        buffered_per_tensor_rows = []

        # Stationarity-driven anneal-then-stop
        controller = CoherenceController(
            window=int(self.config_params.get("controller_window", 500)),
            q=int(self.config_params.get("controller_q", 4)),
            max_decays=int(self.config_params.get("controller_max_decays", 3)),
        )
        controller_start = (
            self.config_params["warmup_steps"] if self.task == 0 else 0
        )
        converged_step = None
        controller_event_rows = []
        fit_start_time = time.perf_counter()

        do_cuda_sync = torch.cuda.is_available()

        def _sync():
            if do_cuda_sync:
                torch.cuda.synchronize()

        for step in range(current_task_steps):
            _sync()
            t0 = time.perf_counter()
            c, c_value = self.compute_c(step, tot_steps=current_task_steps)
            batch = batch_generator.get_batch()
            # Refresh the per-task |grad| EMA once per outer step, the running
            # sum is only updated at the task boundary.
            if self._use_ema:
                self._update_ema(batch, self.theta, self.model)
            _sync()
            timer_accum["c"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            new_theta = self.inner_loop(batch, task_id_tensor, c)
            _sync()
            timer_accum["inner"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            output_task, updated_model_params, weights_info = (
                self.get_updated_params(new_theta, batch, c)
            )
            updated_model_params.update(new_theta["model"])
            predictions = func_call(
                self.model, updated_model_params, batch.x_qr
            )

            inner_loss = self.criterion(predictions, batch.y_qr)
            dummy_loss = self._mse_loss(output_task.squeeze(), task_id_tensor)
            l2reg = l2_regularization(updated_model_params)
            loss = inner_loss + 1e-3 * dummy_loss + self._lambda_l2 * l2reg
            _sync()
            timer_accum["outer_fwd"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            if self.model_optimizer is not None:
                self.model_optimizer.zero_grad()
            self.transformer_optimizer.zero_grad()
            loss.backward()

            # Diagnostics are read after backward and before the clip + step
            meta_grad_cosine_sim, meta_grad_dead_frac_mean, meta_grad_norm = (
                self._compute_meta_grad_diagnostics()
            )

            if meta_grad_norm > 0 and step >= controller_start:
                action = controller.update(self._prev_meta_grad)
                if action is not None:
                    if action == controller.DECAY:
                        transformer_scheduler.decay()
                    controller_event_rows.append(
                        _controller_row(
                            self.task,
                            step + 1,
                            action,
                            controller,
                            transformer_scheduler,
                            current_task_steps,
                            fit_start_time,
                        )
                    )
                if action == controller.DECAY:
                    print(
                        f"Task {int(self.task)} step {step + 1}: stationarity "
                        f"detected ({controller.last_neg_votes}/{controller.q} "
                        f"negative votes, last coherence "
                        f"{controller.last_coherence:.3e}), outer lr decayed "
                        f"to factor {transformer_scheduler.decay_factor:.4f}"
                    )
                elif action == controller.STOP:
                    converged_step = step + 1
                    print(
                        f"Task {int(self.task)} stopped at step {converged_step}: "
                        f"stationarity persists after {controller.n_decays} lr "
                        f"decays ({controller.last_neg_votes}/{controller.q} "
                        f"negative votes, last coherence "
                        f"{controller.last_coherence:.3e})"
                    )

            torch.nn.utils.clip_grad_norm_(self._main_model_parameters, 2.0)
            torch.nn.utils.clip_grad_norm_(
                self._main_transformer_parameters, 1.0
            )

            if self.model_optimizer is not None:
                self.model_optimizer.step()
            self.transformer_optimizer.step()
            _sync()
            timer_accum["outer_bwd"] += time.perf_counter() - t0

            if (step + 1) % log_interval == 0:
                window_steps = (step + 1) - timer_window_start
                step_row, per_tensor_rows, timer_ms = self._build_log_records(
                    step,
                    loss,
                    inner_loss,
                    dummy_loss,
                    l2reg,
                    c_value,
                    weights_info,
                    timer_accum,
                    window_steps,
                    meta_grad_cosine_sim,
                    meta_grad_dead_frac_mean,
                    meta_grad_norm,
                )

                if print_output:
                    self._print_log(
                        step_row, per_tensor_rows, timer_ms, window_steps
                    )

                step_row["lr_decay_factor"] = (
                    transformer_scheduler.decay_factor
                )
                step_row["ctrl_n_decays"] = controller.n_decays
                step_row["ctrl_last_coherence"] = (
                    controller.last_coherence
                    if controller.last_coherence is not None
                    else ""
                )
                step_row["ctrl_last_neg_votes"] = (
                    controller.last_neg_votes
                    if controller.last_neg_votes is not None
                    else ""
                )
                buffered_step_rows.append(step_row)
                buffered_per_tensor_rows.extend(per_tensor_rows)

                for k in timer_accum:
                    timer_accum[k] = 0.0
                timer_window_start = step + 1

            if (
                (step + 1) % checkpoint_interval == 0
                or (step + 1) % current_task_steps == 0
                or converged_step is not None
            ):
                if save_dir is not None:
                    checkpoint = {
                        "step": step + 1,
                        "model_state_dict": self.model.state_dict(),
                        "transformer_state_dict": self.transformer_model.state_dict(),
                        "transformer_optimizer_state_dict": self.transformer_optimizer.state_dict(),
                    }
                    if self.model_optimizer is not None:
                        checkpoint["model_optimizer_state_dict"] = (
                            self.model_optimizer.state_dict()
                        )
                    torch.save(
                        checkpoint,
                        save_dir
                        / f"model_checkpoint_task{int(self.task)}_step{step + 1}.pth",
                    )

            if converged_step is not None:
                break

            transformer_scheduler.step()

        _append_csv_rows(
            metrics_steps_path, _METRICS_STEPS_FIELDS, buffered_step_rows
        )
        _append_csv_rows(
            metrics_per_tensor_path,
            _METRICS_PER_TENSOR_FIELDS,
            buffered_per_tensor_rows,
        )
        if save_dir is not None:
            steps_run = (
                converged_step
                if converged_step is not None
                else current_task_steps
            )
            controller_event_rows.append(
                _controller_row(
                    self.task,
                    steps_run,
                    "converged"
                    if converged_step is not None
                    else "budget_cap",
                    controller,
                    transformer_scheduler,
                    current_task_steps,
                    fit_start_time,
                )
            )
            _append_csv_rows(
                save_dir / "controller.csv",
                _CONTROLLER_FIELDS,
                controller_event_rows,
            )

        self.previous_tsk_score = weights_info["score"]

        if self._use_ema:
            task_importance = self._minmax(self._ema)
            with torch.no_grad():
                for name, importance in (task_importance or {}).items():
                    if name in self._consolidated:
                        self._consolidated[name].add_(importance)
                    else:
                        self._consolidated[name] = importance.clone()
            self._consolidated_norm = None

        weights_info["converged_step"] = converged_step
        return self, weights_info

    def adapt_and_evaluate(
        self, batch, task_id, device=None, print_output=False
    ):
        if device is None:
            device = self._device
        task_idx = torch.tensor(task_id, dtype=torch.float).to(device)
        eval_model = copy.deepcopy(self.model).to(device)
        eval_transformer = copy.deepcopy(self.transformer_model).to(device)

        theta = {
            "model": {
                n: w
                for n, w in eval_model.named_parameters()
                if n not in self._transformer_managed_names
            },
            "transformer": {
                n: w
                for n, w in eval_transformer.named_parameters()
                if w.requires_grad
            },
        }
        optimizer = torch.optim.SGD(
            [
                {"params": theta["model"].values()},
                {"params": theta["transformer"].values()},
            ],
            lr=self._lr_inner,
            momentum=self.config_params["momentum"],
        )

        test_loss = []
        test_accuracy = []
        for step in range(self._test_steps):
            optimizer.zero_grad()
            predictions = func_call(eval_model, None, batch.x_qr)
            loss = self.criterion(predictions, batch.y_qr)
            acc = accuracy(predictions, batch.y_qr)
            test_accuracy.append(acc)
            test_loss.append(loss.item())

            output_task, updated_model_params, weights_info = (
                self.get_updated_params(
                    theta,
                    batch,
                    c=None,
                    model=eval_model,
                    transformer_model=eval_transformer,
                    device=device,
                )
            )
            predictions = func_call(
                eval_model, updated_model_params, batch.x_sp
            )
            inner_loss = self.criterion(predictions, batch.y_sp)
            dummy_loss = self._mse_loss(output_task.squeeze(), task_idx)
            l2reg = l2_regularization(updated_model_params)
            loss = inner_loss + 1e-3 * dummy_loss + self._lambda_l2 * l2reg
            loss.backward()
            optimizer.step()

        if print_output:
            print(
                "\nEvaluation: Final loss: {:.4f}, Final accuracy: {:.3f}%\n".format(
                    test_loss[-1], 100.0 * test_accuracy[-1]
                )
            )
        return test_accuracy, test_loss

    def evaluate(self, data, task_id, nb_test_batch=5, device=None):
        test_generator = self._batch_generator_class(data, self.config_params)
        avg_accuracy = []
        for _ in range(nb_test_batch):
            batch = test_generator.get_batch(device)
            test_accuracy, _ = self.adapt_and_evaluate(batch, task_id, device)
            avg_accuracy.append(test_accuracy)
        avg_accuracy = np.array(avg_accuracy).mean(axis=0)
        return avg_accuracy[-1]
