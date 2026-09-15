"""Parameter selection and block counts shared by the optimizer and experiments."""

import math

DEFAULT_MIN_INIT_STD = 1e-6


def _segments_contain(parts, token_parts):
    span = len(token_parts)
    for start in range(len(parts) - span + 1):
        if parts[start : start + span] == token_parts:
            return True
    return False


def _name_matches_any_token(name, tokens):
    parts = name.split(".")
    for token in tokens:
        if "." in token:
            # Match the token as a run of dotted segments anywhere in the
            # name, so a wrapped backbone (e.g. L2P/DyTox "backbone." prefix)
            # is matched by the same unwrapped token used for the bare ViT.
            if _segments_contain(parts, token.split(".")):
                return True
        elif token in parts:
            return True
    return False


def _is_leaf_module(module):
    children = module.children()
    try:
        next(children)
        return False
    except StopIteration:
        return True


def _matches_filters(name, include_tokens, exclude_tokens):
    if include_tokens and not _name_matches_any_token(name, include_tokens):
        return False
    if exclude_tokens and _name_matches_any_token(name, exclude_tokens):
        return False
    return True


def summarize_train_scope(
    model, *, include_tokens=(), exclude_tokens=(), min_init_std=0.0
):
    """Summarizes the parameters the transformer meta-optimizer will manage.

    include_tokens / exclude_tokens filter parameters by dot-split name.
    min_init_std drops tensors whose init-time std is below the floor, so the
    scope matches the set the meta-optimizer actually updates and the
    transformer output length equals the number of updated weights.
    """
    include_tokens = list(include_tokens)
    exclude_tokens = list(exclude_tokens)

    named_params = list(model.named_parameters())

    selected = []
    filter_matched_param_ids = set()
    for name, p in named_params:
        matches = _matches_filters(name, include_tokens, exclude_tokens)
        if matches:
            filter_matched_param_ids.add(id(p))
        # "not (std < floor)" keeps NaN-std tensors, matching PROPOSED.__init__.
        if matches and not (p.data.std().item() < min_init_std):
            selected.append((name, p))

    selected_tensor_numels = []
    selected_tensor_d0 = []
    selected_param_names = []
    trainable_info = []
    selected_total_params = 0
    trainable_params = 0

    for name, p in selected:
        numel = int(p.numel())
        d0 = int(p.shape[0]) if p.ndim > 0 else 0
        selected_tensor_numels.append(numel)
        selected_tensor_d0.append(d0)
        selected_param_names.append(name)
        selected_total_params += numel
        if p.requires_grad:
            trainable_info.append({"name": name, "numel": numel, "d0": d0})
            trainable_params += numel

    trainable_param_names = [t["name"] for t in trainable_info]

    # A leaf qualifies when it owns at least one trainable direct parameter
    # and at least one direct parameter matching the filters; these need not
    # be the same tensor.
    trainable_leaf_module_names = []
    for module_name, module in model.named_modules():
        if not _is_leaf_module(module):
            continue
        direct_params = list(module.parameters(recurse=False))
        if not direct_params:
            continue
        if not any(p.requires_grad for p in direct_params):
            continue
        if any(id(p) in filter_matched_param_ids for p in direct_params):
            trainable_leaf_module_names.append(module_name)

    if not trainable_leaf_module_names:
        scope_label = "none"
    elif len(trainable_leaf_module_names) == 1:
        scope_label = "single_layer"
    else:
        scope_label = "multi_layer"

    return {
        "selected_total_parameters": int(selected_total_params),
        "selected_parameter_tensors": int(len(selected)),
        "selected_tensor_numels": selected_tensor_numels,
        "selected_tensor_d0": selected_tensor_d0,
        "selected_param_names": selected_param_names,
        "trainable_layer_stats": trainable_info,
        "trainable_parameters": int(trainable_params),
        "trainable_parameter_tensors": int(len(trainable_info)),
        "trainable_param_names": trainable_param_names,
        "include_tokens": include_tokens,
        "exclude_tokens": exclude_tokens,
        "trainable_leaf_module_count": int(len(trainable_leaf_module_names)),
        "trainable_leaf_module_names": trainable_leaf_module_names,
        "training_scope_label": scope_label,
        "last_trainable_module": trainable_leaf_module_names[-1]
        if trainable_leaf_module_names
        else None,
    }


def calculate_transformer_blocks(scope_summary, config):
    """Computes block partitioning stats, mirroring TransformerModel exactly."""
    strategy = config.get("block_strategy", "row_wise")
    layers = scope_summary["trainable_layer_stats"]

    blocks_per_param = {}
    max_blocks = 0
    total_blocks = 0

    if strategy == "flatten":
        alpha = float(config["alpha"])
        beta = float(config["beta"])
        for layer in layers:
            name = layer["name"]
            numel = layer["numel"]
            if numel == 0:
                continue
            block_size = max(1, int(numel**alpha / beta))
            n_blocks = int(math.ceil(numel / block_size))
            blocks_per_param[name] = n_blocks
            total_blocks += n_blocks
            if n_blocks > max_blocks:
                max_blocks = n_blocks
    else:
        block_rows = int(config.get("block_rows", 4))
        row_scale = float(config.get("row_scale", 1.0))
        for layer in layers:
            name = layer["name"]
            numel = layer["numel"]
            d0 = layer["d0"]
            if numel == 0:
                continue
            d0_eff = max(1, int(d0 * row_scale))
            row_len = max(1, numel // d0_eff)
            d0_eff = numel // row_len
            n_blocks = int(math.ceil(d0_eff / block_rows))
            blocks_per_param[name] = n_blocks
            total_blocks += n_blocks
            if n_blocks > max_blocks:
                max_blocks = n_blocks

    return {
        "total_blocks": total_blocks,
        "max_blocks_in_any_tensor": max_blocks,
        "blocks_per_parameter_tensor": blocks_per_param,
    }
