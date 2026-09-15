"""Block-level transformer meta-optimizer for continual learning.
Defaults for every config flag are in __init__.
"""

import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .parameter_scope import calculate_transformer_blocks
from .task_encoder import TaskEncoderGeneric

# log_fan_in, log_fan_out, log_param_count, relative_depth, block_position.
BLOCK_SIGNATURE_DIM = 5


class TransformerModel(nn.Module):
    def __init__(self, config_params, base_model_info, task_encoder=None):
        super().__init__()

        self.d_model = config_params["embedding_size"]
        self.nhead = config_params["num_heads"]
        self.d_hid = config_params["dim_feedforward"]
        self.nlayers = config_params["n_layers"]
        self.dropout = config_params.get("dropout", 0.0)
        self.clamp = config_params.get("clamping_transformer", 1.0)
        self.device = config_params["device"]

        self.block_strategy = config_params.get("block_strategy", "row_wise")
        self.block_rows = int(config_params.get("block_rows") or 4)
        self.row_scale = float(config_params.get("row_scale") or 1.0)
        self.importance_eps = float(config_params.get("importance_eps", 1e-3))

        self.use_layer_id = bool(config_params.get("use_layer_id", True))
        self.use_block_pos_embedding = bool(
            config_params.get("use_block_pos_embedding", self.use_layer_id)
        )
        self.use_weight_stats = bool(
            config_params.get("use_weight_stats", True)
        )
        self.use_stability = bool(config_params.get("use_stability", True))

        self.use_tensor_embedding = bool(
            config_params.get("use_tensor_embedding", False)
        )
        self.tensor_embed_dim = int(config_params.get("tensor_embed_dim", 8))

        self.use_block_signature = bool(
            config_params.get("use_block_signature", False)
        )

        self.use_signed_gradient_basis = bool(
            config_params.get("use_signed_gradient_basis", False)
        )

        self.use_pos_encoder = bool(
            config_params.get("use_pos_encoder", False)
        )

        # The continuous magnitude head for signed-gradient basis
        plasticity_scales_cfg = config_params.get("plasticity_scales", None)
        if (
            self.use_signed_gradient_basis
            and plasticity_scales_cfg is not None
            and min(plasticity_scales_cfg) <= 0.0
        ):
            raise ValueError(
                "The continuous magnitude head requires every plasticity_scales "
                f"entry to be strictly positive. Got plasticity_scales="
                f"{list(plasticity_scales_cfg)}."
            )

        self._validate_config()

        if plasticity_scales_cfg is not None:
            self.register_buffer(
                "plasticity_scales",
                torch.tensor(plasticity_scales_cfg, dtype=torch.float32),
            )
        else:
            self.plasticity_scales = None

        self.nweights = int(base_model_info["selected_total_parameters"])
        n_model_layers = int(
            base_model_info.get("selected_parameter_tensors", 1)
        )

        total_blocks, max_blocks_per_layer = self._init_block_sizes(
            config_params, base_model_info
        )

        if task_encoder is None:
            self.te_model = TaskEncoderGeneric(
                config_params
            ).to(self.device)
        else:
            self.te_model = task_encoder.to(self.device)

        encoder_layers = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_hid,
            dropout=self.dropout,
            activation="gelu",
            batch_first=False,
        ).to(self.device)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers=self.nlayers
        ).to(self.device)

        # Per-block input dim. Order: importance(4), weight_stats(4)?,
        # block_signature(5)?, tensor_embedding(D)? -> tbd ablation
        in_dim = 4  # mean_imp_all, max_imp_all, mean_imp_pos, density_imp
        if self.use_weight_stats:
            in_dim += 4  # weight mean, std, abs.mean, abs.max
        if self.use_block_signature:
            in_dim += BLOCK_SIGNATURE_DIM
        if self.use_tensor_embedding:
            in_dim += self.tensor_embed_dim

        self.embedding = FeatureExtractor(in_dim, self.d_model - 1).to(
            self.device
        )
        if self.use_pos_encoder:
            self.pos_encoder = PositionalEncoding(
                max_seq_len=1 + total_blocks,
                d_model=self.d_model,
            ).to(self.device)

        if self.use_tensor_embedding:
            self.tensor_embedding = nn.Embedding(
                n_model_layers + 1, self.tensor_embed_dim
            ).to(self.device)

        if self.use_layer_id:
            self.layer_embedding = nn.Embedding(
                n_model_layers + 1, self.d_model
            ).to(self.device)
        if self.use_block_pos_embedding:
            self.block_pos_embedding = nn.Embedding(
                max_blocks_per_layer + 1, self.d_model
            ).to(self.device)

        self.linear_task = nn.Linear(self.d_model, 1).to(self.device)
        # The signed-gradient basis predicts K=3 coefficients per block
        # the scatter mixes them with element-level basis signals.
        basis_output_dim = 3 if self.use_signed_gradient_basis else 1
        self.linear_blocks = nn.Linear(self.d_model, basis_output_dim).to(
            self.device
        )

        self.tanh = nn.Tanh()
        self.init_weights()

        # Layout cache, populated lazily on first forward.
        self._block_cache = None

    def _validate_config(self):
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"embedding_size ({self.d_model}) must be divisible by num_heads ({self.nhead})"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if self.clamp <= 0:
            raise ValueError("clamping_transformer must be > 0")
        if self.block_strategy not in ("row_wise", "flatten"):
            raise ValueError("block_strategy must be 'row_wise' or 'flatten'")

    def _init_block_sizes(self, config_params, base_model_info):
        """Returns (total_blocks, max_blocks_per_layer) for sizing embeddings."""
        if self.block_strategy == "flatten":
            self.alpha = float(config_params["alpha"])
            self.beta = float(config_params["beta"])
            tensor_numels = list(base_model_info["selected_tensor_numels"])
            self.block_size_per_tensor = [
                max(1, int(n**self.alpha / self.beta)) for n in tensor_numels
            ]

        block_info = calculate_transformer_blocks(
            base_model_info, config_params
        )
        return block_info["total_blocks"], block_info[
            "max_blocks_in_any_tensor"
        ]

    def init_weights(self):
        initrange = 0.1
        for linear in (self.linear_task, self.linear_blocks):
            linear.bias.data.zero_()
            linear.weight.data.uniform_(-initrange, initrange)

    def _apply_eps(self, scores):
        if self.importance_eps > 0.0:
            return scores.masked_fill(scores < self.importance_eps, 0.0)
        return scores

    @staticmethod
    def _fan_in_out(shape):
        """Fan-in / fan-out from a weight tensor's shape.

        Conv2d [out, in, kH, kW] uses receptive-field-weighted fans, Linear
        [out, in] the definition. Anything else is numel
        for both, which gives a finite log-scaled signal.
        """
        if len(shape) == 4:
            out_ch, in_ch, kh, kw = shape
            return in_ch * kh * kw, out_ch * kh * kw
        if len(shape) == 2:
            out_f, in_f = shape
            return in_f, out_f
        if len(shape) == 1:
            return shape[0], shape[0]
        numel = 1
        for s in shape:
            numel *= s
        return numel, numel

    def _build_layout(self, weights):
        """Precomputes per-block gather indices and per-tensor anchors.

        All blocks live as rows of a [num_blocks, max_block_size] panel. Real
        entries are addressed by global_indices. Padded entries point to a
        scratch slot zeroed before reductions.
        """
        names = []
        tensor_offsets = {}
        tensor_numels = []
        tensor_shapes = []
        running = 0
        for name, weight in weights.items():
            if weight.numel() == 0:
                continue
            names.append(name)
            tensor_offsets[name] = running
            tensor_numels.append(weight.numel())
            tensor_shapes.append(tuple(weight.shape))
            running += weight.numel()
        n_total = running

        slices = []
        blocks_per_tensor = []
        for tensor_idx, name in enumerate(names):
            numel = tensor_numels[tensor_idx]
            weight = weights[name]
            if self.block_strategy == "flatten":
                block_size = self.block_size_per_tensor[tensor_idx]
                bounds = [
                    (start, min(start + block_size, numel))
                    for start in range(0, numel, block_size)
                ]
            else:
                d0_eff = max(1, int(weight.shape[0] * self.row_scale))
                row_len = max(1, numel // d0_eff)
                d0_eff = numel // row_len
                bounds = [
                    (
                        row_start * row_len,
                        min(row_start + self.block_rows, d0_eff) * row_len,
                    )
                    for row_start in range(0, d0_eff, self.block_rows)
                ]
            blocks_per_tensor.append(len(bounds))
            for block_idx, (lo, hi) in enumerate(bounds):
                slices.append((tensor_idx, name, lo, hi, block_idx))

        if not slices:
            return None

        num_blocks = len(slices)
        max_block_size = max(hi - lo for _, _, lo, hi, _ in slices)

        global_indices = torch.full(
            (num_blocks, max_block_size),
            fill_value=n_total,  # sentinel: points at the zeroed scratch slot
            dtype=torch.long,
            device=self.device,
        )
        valid_mask = torch.zeros(
            (num_blocks, max_block_size), dtype=torch.bool, device=self.device
        )
        layer_ids = torch.zeros(
            num_blocks, dtype=torch.long, device=self.device
        )
        block_pos_ids = torch.zeros(
            num_blocks, dtype=torch.long, device=self.device
        )
        block_sizes = torch.zeros(
            num_blocks, dtype=torch.long, device=self.device
        )

        flat_indices_parts = []
        for b, (tensor_idx, name, lo, hi, block_idx) in enumerate(slices):
            size = hi - lo
            start = tensor_offsets[name] + lo
            arange = torch.arange(
                start, start + size, device=self.device, dtype=torch.long
            )
            global_indices[b, :size] = arange
            valid_mask[b, :size] = True
            layer_ids[b] = tensor_idx
            block_pos_ids[b] = block_idx
            block_sizes[b] = size
            flat_indices_parts.append(arange)

        flat_indices = (
            torch.cat(flat_indices_parts)
            if flat_indices_parts
            else torch.empty(0, dtype=torch.long, device=self.device)
        )
        block_ids_per_element = torch.arange(
            num_blocks, device=self.device
        ).repeat_interleave(block_sizes)

        cache = {
            "names": names,
            "tensor_offsets": tensor_offsets,
            "tensor_numels": tensor_numels,
            "tensor_shapes": tensor_shapes,
            "blocks_per_tensor": blocks_per_tensor,
            "num_blocks": num_blocks,
            "global_indices": global_indices,
            "valid_mask": valid_mask,
            "valid_mask_f": valid_mask.float(),
            "layer_ids": layer_ids,
            "block_pos_ids": block_pos_ids,
            "block_sizes": block_sizes,
            "block_sizes_f": block_sizes.float().clamp(min=1.0),
            "flat_indices": flat_indices,
            "block_ids_per_element": block_ids_per_element,
            "dense_output": flat_indices.numel() == self.nweights,
        }

        if self.use_block_signature:
            cache["block_signature"] = self._compute_block_signature(cache)

        return cache

    def _compute_block_signature(self, cache):
        """Per-block intrinsic descriptor of shape [num_blocks, 5].

        Every column is either a per-tensor scalar broadcast to that tensor's
        blocks or a simple per-block function of the layout, so the whole
        signature is computed vectorised at layout time.
        """
        num_blocks = cache["num_blocks"]
        names = cache["names"]
        shapes = cache["tensor_shapes"]
        blocks_per_tensor = cache["blocks_per_tensor"]
        n_tensors = len(names)
        depth_denom = max(1, n_tensors - 1)

        fan_in_per_tensor = []
        fan_out_per_tensor = []
        for tensor_idx, name in enumerate(names):
            fan_in, fan_out = self._fan_in_out(shapes[tensor_idx])
            fan_in_per_tensor.append(float(fan_in))
            fan_out_per_tensor.append(float(fan_out))

        device = self.device
        dtype = torch.float32

        fan_in_t = torch.tensor(fan_in_per_tensor, device=device, dtype=dtype)
        fan_out_t = torch.tensor(
            fan_out_per_tensor, device=device, dtype=dtype
        )
        blocks_per_tensor_t = torch.tensor(
            blocks_per_tensor, device=device, dtype=dtype
        )

        layer_ids = cache["layer_ids"]
        block_pos_ids = cache["block_pos_ids"]
        block_sizes_f = cache["block_sizes"].to(dtype)

        log_fan_in = torch.log1p(fan_in_t.index_select(0, layer_ids))
        log_fan_out = torch.log1p(fan_out_t.index_select(0, layer_ids))

        relative_depth = layer_ids.to(dtype) / float(depth_denom)

        # block_position normalises within each tensor.
        position_denom = (
            blocks_per_tensor_t.index_select(0, layer_ids) - 1.0
        ).clamp(min=1.0)
        block_position = block_pos_ids.to(dtype) / position_denom

        log_param_count = torch.log1p(block_sizes_f)

        signature = torch.empty(
            num_blocks, BLOCK_SIGNATURE_DIM, device=device, dtype=dtype
        )
        signature[:, 0] = log_fan_in
        signature[:, 1] = log_fan_out
        signature[:, 2] = log_param_count
        signature[:, 3] = relative_depth
        signature[:, 4] = block_position
        return signature

    def _gather_blocks(self, flat_all, cache):
        """Gathers a flat [N_total] tensor into [num_blocks, max_block_size].

        Padded slots map to a scratch 0 appended at the end, so reductions
        over masked entries are safe even before the mask is applied.
        """
        scratch = flat_all.new_zeros(1)
        extended = torch.cat([flat_all, scratch], dim=0)
        return extended[cache["global_indices"]]

    def _scatter_with_basis(
        self,
        block_alphas,
        weights,
        importance_scores,
        raw_gradients,
        init_std_map,
        device,
    ):
        """Element-level updates from per-block alphas dotted with 3 basis signals.

        Per element i in block b:
          update[i] = alpha_0[b]
                    + alpha_1[b] * w[i] / init_std_of_tensor
                    + alpha_2[b] * g[i] / g_scale_of_block

        Both non-constant bases are scale-invariant, so one alpha magnitude
        means the same thing across heterogeneous tensors, and SGD descent is
        recoverable as alpha = (0, 0, -1). init_std is static per tensor;
        g_scale is the per-block mean |g|, recomputed each forward because the
        gradient depends on the batch.
        """
        if raw_gradients is None or init_std_map is None:
            raise ValueError(
                "signed-gradient basis requires raw_gradients and init_std_map"
            )

        cache = self._block_cache
        flat_indices = cache["flat_indices"]
        block_sizes = cache["block_sizes"]
        block_sizes_f = cache["block_sizes_f"]
        block_ids_per_element = cache["block_ids_per_element"]
        num_blocks = block_sizes.numel()

        flat_weight_parts = []
        flat_gradient_parts = []
        for name in cache["names"]:
            flat_weight_parts.append(weights[name].reshape(-1))
            flat_gradient_parts.append(
                raw_gradients[name].reshape(-1).detach()
            )
        flat_weight = torch.cat(flat_weight_parts)
        flat_gradient = torch.cat(flat_gradient_parts)

        weight_block_order = flat_weight[flat_indices]
        gradient_block_order = flat_gradient[flat_indices]

        # init_std per element depends only on init_std_map and the fixed
        # layout, so it is built once and cached.
        init_std_block_order = cache.get("init_std_block_order")
        if init_std_block_order is None:
            init_std_per_element = torch.cat(
                [
                    torch.full_like(
                        weights[name].reshape(-1), float(init_std_map[name])
                    )
                    for name in cache["names"]
                ]
            )
            init_std_block_order = init_std_per_element[flat_indices]
            cache["init_std_block_order"] = init_std_block_order

        g_scale_per_block = torch.zeros(num_blocks, device=device)
        g_scale_per_block.scatter_add_(
            0, block_ids_per_element, gradient_block_order.abs()
        )
        g_scale_per_block = (g_scale_per_block / block_sizes_f).clamp(min=1e-6)
        g_scale_per_element = g_scale_per_block.index_select(
            0, block_ids_per_element
        )

        basis_constant = torch.ones_like(weight_block_order)
        basis_weight = (weight_block_order / init_std_block_order).clamp(
            -3.0, 3.0
        )
        basis_gradient = (gradient_block_order / g_scale_per_element).clamp(
            -3.0, 3.0
        )
        basis_per_element = torch.stack(
            [basis_constant, basis_weight, basis_gradient], dim=1
        )

        alphas_per_element = block_alphas.index_select(
            0, block_ids_per_element
        )
        update_block_order = (alphas_per_element * basis_per_element).sum(
            dim=1
        )

        flat_score = torch.cat(
            [importance_scores[name].reshape(-1) for name in cache["names"]]
        )
        keep = (flat_score[flat_indices] > 0).to(update_block_order.dtype)
        update_block_order = update_block_order * keep

        if cache["dense_output"]:
            return update_block_order

        output_weights = torch.zeros(self.nweights, device=device)
        output_weights[flat_indices] = update_block_order
        return output_weights

    def _build_blocks(
        self, weights, importance_scores, prev_importance_scores=None
    ):
        if self._block_cache is None:
            built = self._build_layout(weights)
            if built is None:
                return None
            self._block_cache = built
        cache = self._block_cache

        weight_parts = [] if self.use_weight_stats else None
        importance_parts = []
        prev_importance_parts = [] if self.use_stability else None
        for name in cache["names"]:
            if self.use_weight_stats:
                weight_parts.append(weights[name].reshape(-1))
            importance_parts.append(
                self._apply_eps(importance_scores[name].reshape(-1))
            )
            if self.use_stability:
                if (
                    prev_importance_scores is not None
                    and name in prev_importance_scores
                ):
                    prev_importance_parts.append(
                        self._apply_eps(
                            prev_importance_scores[name].reshape(-1)
                        )
                    )
                else:
                    prev_importance_parts.append(
                        torch.zeros_like(weights[name].reshape(-1))
                    )

        flat_importance = torch.cat(importance_parts)
        block_importance = self._gather_blocks(flat_importance, cache)

        if self.use_weight_stats:
            flat_weights = torch.cat(weight_parts)
            block_weights = self._gather_blocks(flat_weights, cache)

        if self.use_stability:
            flat_prev_importance = torch.cat(prev_importance_parts)
            block_prev_importance = self._gather_blocks(
                flat_prev_importance, cache
            )

        mask_f = cache["valid_mask_f"]
        mask_bool = cache["valid_mask"]
        block_sizes_f = cache["block_sizes_f"]

        mean_imp_all = (block_importance * mask_f).sum(dim=1) / block_sizes_f
        importance_for_max = block_importance.masked_fill(
            ~mask_bool, float("-inf")
        )
        max_imp_all = importance_for_max.max(dim=1).values
        max_imp_all = max_imp_all.masked_fill(torch.isinf(max_imp_all), 0.0)

        pos_mask = (block_importance > 0) & mask_bool
        pos_mask_f = pos_mask.float()
        pos_count = pos_mask_f.sum(dim=1)
        density_imp = pos_count / block_sizes_f

        pos_sum = (block_importance * pos_mask_f).sum(dim=1)
        safe_pos_count = pos_count.clamp(min=1.0)
        mean_imp_pos = torch.where(
            pos_count > 0,
            pos_sum / safe_pos_count,
            torch.zeros_like(pos_sum),
        )

        layer_ids = cache["layer_ids"]
        n_tensors = len(cache["names"])
        per_tensor_max = torch.zeros(n_tensors, device=mean_imp_pos.device)
        per_tensor_max.scatter_reduce_(
            0, layer_ids, mean_imp_pos, reduce="amax", include_self=False
        )
        per_tensor_max = per_tensor_max.clamp(min=1e-8)
        mean_imp_pos_local = mean_imp_pos / per_tensor_max[layer_ids]

        plasticity = torch.sqrt(
            (mean_imp_pos_local * density_imp).clamp_min(0.0)
        )

        if self.use_stability:
            prev_pos_mask = (block_prev_importance > 0) & mask_bool
            prev_pos_mask_f = prev_pos_mask.float()
            prev_pos_count = prev_pos_mask_f.sum(dim=1)
            prev_density = prev_pos_count / block_sizes_f
            prev_pos_sum = (block_prev_importance * prev_pos_mask_f).sum(dim=1)
            safe_prev_pos_count = prev_pos_count.clamp(min=1.0)
            mean_imp_pos_prev = torch.where(
                prev_pos_count > 0,
                prev_pos_sum / safe_prev_pos_count,
                torch.zeros_like(prev_pos_sum),
            )
            stability = torch.sqrt(
                (mean_imp_pos_prev * prev_density).clamp_min(0.0)
            )
            effective = torch.sqrt((plasticity * (1.0 - stability)))
        else:
            effective = plasticity

        feature_columns = [
            mean_imp_all,
            max_imp_all,
            mean_imp_pos,
            density_imp,
        ]

        if self.use_weight_stats:
            block_weights_masked = block_weights * mask_f
            w_mean = block_weights_masked.sum(dim=1) / block_sizes_f
            # Bessel-corrected std, matching torch's .std() default.
            diff = (block_weights - w_mean.unsqueeze(1)) * mask_f
            n_minus_1 = (block_sizes_f - 1.0).clamp(min=1.0)
            w_var = (diff * diff).sum(dim=1) / n_minus_1
            w_std = w_var.sqrt()
            w_abs_mean = block_weights_masked.abs().sum(dim=1) / block_sizes_f
            weights_abs_for_max = block_weights.abs().masked_fill(
                ~mask_bool, float("-inf")
            )
            w_abs_max = weights_abs_for_max.max(dim=1).values
            w_abs_max = w_abs_max.masked_fill(torch.isinf(w_abs_max), 0.0)
            feature_columns += [w_mean, w_std, w_abs_mean, w_abs_max]

        block_stats = torch.stack(feature_columns, dim=1)

        if self.use_block_signature:
            block_stats = torch.cat(
                [block_stats, cache["block_signature"]], dim=1
            )

        block_scores = plasticity

        plasticity_level_ids = None
        if (
            self.plasticity_scales is not None
            and not self.use_signed_gradient_basis
        ):
            n_levels = self.plasticity_scales.numel()
            effective_clamped = effective.clamp(0.0, 1.0)
            plasticity_level_ids = (
                torch.floor(effective_clamped * n_levels)
                .long()
                .clamp(0, n_levels - 1)
            )

        return (
            block_stats,
            block_scores,
            plasticity_level_ids,
            effective,  # Read by the continuous magnitude branch in forward.
            cache["layer_ids"],
            cache["block_pos_ids"],
        )

    def forward(
        self,
        x,
        weights,
        importance_scores,
        src_mask=None,
        device=None,
        prev_importance_scores=None,
        raw_gradients=None,
        init_std_map=None,
    ):
        device = device or self.device

        task_encoding = self.te_model(x.to(device))
        if task_encoding.dim() == 1:
            task_encoding = task_encoding.unsqueeze(0)

        if task_encoding.size(-1) != self.d_model:
            raise ValueError(
                f"Task encoder output dim {task_encoding.size(-1)} != d_model {self.d_model}"
            )

        result = self._build_blocks(
            weights, importance_scores, prev_importance_scores
        )
        if result is None:
            return self.linear_task(task_encoding), torch.zeros(
                self.nweights, device=device
            )

        (
            block_stats,
            block_scores,
            plasticity_level_ids,
            effective,
            layer_ids,
            block_pos_ids,
        ) = result

        if self.use_tensor_embedding:
            tensor_emb = self.tensor_embedding(layer_ids)
            block_stats = torch.cat([block_stats, tensor_emb], dim=1)

        block_feats = self.embedding(block_stats)

        block_tokens = torch.cat(
            [block_feats, block_scores.unsqueeze(1)], dim=1
        )

        if self.use_layer_id:
            block_tokens = block_tokens + self.layer_embedding(layer_ids)
        if self.use_block_pos_embedding:
            block_tokens = block_tokens + self.block_pos_embedding(block_pos_ids)

        src = torch.cat([task_encoding, block_tokens], dim=0)
        if self.use_pos_encoder:
            src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)

        output_task = self.linear_task(output[: task_encoding.shape[0]])
        block_output = output[task_encoding.shape[0] :]

        block_head_output = (
            self.tanh(self.linear_blocks(block_output)) * self.clamp
        )

        if self.use_signed_gradient_basis:
            if self.plasticity_scales is not None:
                min_s = self.plasticity_scales.min()
                max_s = self.plasticity_scales.max()
                log_span = max_s.log() - min_s.log()
                effective_clamped = effective.clamp(0.0, 1.0)

                raw_magnitude = torch.exp(
                    min_s.log() + effective_clamped * log_span
                )

                magnitude = (raw_magnitude - min_s) / (max_s - min_s).clamp(
                    min=1e-12
                )
                magnitude = magnitude * max_s

                block_head_output = block_head_output * magnitude.unsqueeze(1)
            output_weights = self._scatter_with_basis(
                block_alphas=block_head_output,
                weights=weights,
                importance_scores=importance_scores,
                raw_gradients=raw_gradients,
                init_std_map=init_std_map,
                device=device,
            )
        else:
            block_updates = block_head_output.squeeze(1)
            if plasticity_level_ids is not None:
                block_updates = (
                    block_updates
                    * self.plasticity_scales[plasticity_level_ids]
                )
            cache = self._block_cache
            repeated_updates = block_updates.repeat_interleave(
                cache["block_sizes"]
            )
            if cache["dense_output"]:
                output_weights = repeated_updates
            else:
                output_weights = torch.zeros(self.nweights, device=device)
                output_weights[cache["flat_indices"]] = repeated_updates

        return output_task, output_weights


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.empty(max_seq_len, d_model).normal_(std=0.2)
        )

    def forward(self, x):
        return x + self.pos_embedding[: x.size(0)]


class FeatureExtractor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
