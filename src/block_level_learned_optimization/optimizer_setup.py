"""Construct the base-model and learned-optimizer optimizers."""

import torch

from .parameter_scope import _matches_filters


def set_optimizer(
    model,
    transformer_model,
    config_params,
    include_tokens=None,
    exclude_tokens=(),
):
    """Returns (model_optimizer, transformer_optimizer).

    Parameters matching include_tokens are transformer-managed and excluded
    from the base Adam optimizer. include_tokens=None falls back to
    model.pred_with_transformer. Returns (None, transformer_optimizer) when
    no base parameters remain for Adam.
    """
    if include_tokens is None:
        include_tokens = list(getattr(model, "pred_with_transformer", []))
    else:
        include_tokens = list(include_tokens)
    exclude_tokens = list(exclude_tokens)

    has_include = bool(include_tokens)
    base_params = [
        p
        for name, p in model.named_parameters()
        if not (
            has_include
            and _matches_filters(name, include_tokens, exclude_tokens)
        )
    ]

    model_optimizer = None
    if base_params:
        model_optimizer = torch.optim.Adam(
            base_params,
            lr=config_params["lr"],
            weight_decay=config_params["lambda_l2"],
        )
    transformer_optimizer = make_transformer_optimizer(
        transformer_model, config_params
    )
    return model_optimizer, transformer_optimizer


def make_transformer_optimizer(transformer_model, config_params):
    """Builds the outer (meta) optimizer: "adam" (default) or the
    learning-rate-free "prodigy" ablation (Mishchenko & Defazio, 2024)."""
    name = str(config_params.get("outer_optimizer", "adam")).lower()
    lambda_l2 = config_params["lambda_l2"]
    if name == "adam":
        return torch.optim.Adam(
            transformer_model.parameters(),
            lr=config_params.get("lr_transformer", config_params["lr"]),
            weight_decay=lambda_l2,
        )
    if name == "prodigy":
        try:
            from prodigyopt import Prodigy
        except ImportError as e:
            raise ImportError(
                "outer_optimizer='prodigy' requires `pip install prodigyopt`"
            ) from e
        return Prodigy(
            transformer_model.parameters(),
            lr=1.0,
            weight_decay=lambda_l2,
        )
    raise ValueError(f"Unknown outer_optimizer: {name!r}")
