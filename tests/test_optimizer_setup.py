import torch
from torch import nn

from block_level_learned_optimization.optimizer_setup import set_optimizer
from block_level_learned_optimization.parameter_scope import (
    DEFAULT_MIN_INIT_STD,
    summarize_train_scope,
)


def test_adam_updates_tensors_below_initial_std_threshold():
    model = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2), nn.Linear(2, 2))
    model.pred_with_transformer = ["0", "1"]
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[-1.0, 1.0], [1.0, -1.0]]))
        model[0].bias.zero_()
    transformer = nn.Linear(2, 2)
    config = {"lr": 0.01, "lambda_l2": 0.0}

    model_optimizer, _ = set_optimizer(model, transformer, config)
    adam_parameter_ids = {
        id(parameter) for parameter in model_optimizer.param_groups[0]["params"]
    }
    expected_adam_parameter_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name != "0.weight"
    }
    assert adam_parameter_ids == expected_adam_parameter_ids

    scope = summarize_train_scope(
        model,
        include_tokens=model.pred_with_transformer,
        min_init_std=DEFAULT_MIN_INIT_STD,
    )
    assert scope["selected_param_names"] == ["0.weight"]

    original_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    loss = sum(parameter.sum() for parameter in model.parameters())
    loss.backward()
    model_optimizer.step()

    for name, parameter in model.named_parameters():
        if name == "0.weight":
            assert torch.equal(parameter, original_parameters[name])
        else:
            assert not torch.equal(parameter, original_parameters[name])

    config["min_init_std"] = 2.0
    model_optimizer, _ = set_optimizer(model, transformer, config)
    adam_parameter_ids = {
        id(parameter) for parameter in model_optimizer.param_groups[0]["params"]
    }
    assert adam_parameter_ids == {id(parameter) for parameter in model.parameters()}
