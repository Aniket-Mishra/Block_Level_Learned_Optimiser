"""Check shared transformer weights on new models and a custom task encoder."""

import argparse
import csv
import math
from collections import Counter, OrderedDict
from pathlib import Path

import torch
from torch import nn

import dataset
from exp_runner_1_1_config import ablation_jobs
from exp_runner_1_1_fn_setup import (
    _build_task_encoder,
    _set_reproducibility,
    handler,
)
from block_level_learned_optimization.parameter_scope import summarize_train_scope
from block_level_learned_optimization.task_encoder import LambdaLayer
from block_level_learned_optimization.trainer import PROPOSED
from block_level_learned_optimization.transformer import TransformerModel
from utils import set_device


def make_rgb_encoder(embedding_size):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, embedding_size),
        LambdaLayer(lambda values: values.mean(dim=0)),
    )


def load_shared_weights(transformer, shared_state):
    target_state = transformer.state_dict()
    expected_keys = {
        name for name in target_state if not name.startswith("te_model.")
    }
    if shared_state.keys() != expected_keys:
        missing = sorted(expected_keys - shared_state.keys())
        extra = sorted(shared_state.keys() - expected_keys)
        raise ValueError(f"Shared checkpoint keys differ: missing={missing}, extra={extra}")
    for name, value in shared_state.items():
        if value.shape != target_state[name].shape:
            raise ValueError(
                f"Shared checkpoint shape differs for {name}: "
                f"{tuple(value.shape)} != {tuple(target_state[name].shape)}"
            )
    target_state.update(shared_state)
    transformer.load_state_dict(target_state, strict=True)


def train_source(output_dir):
    job = ablation_jobs(
        "splitmnist", "all_but_head", anchor_only=True,
        include_ema_axis=False,
        flag_overrides={
            "use_layer_id": False,
            "use_tensor_embedding": False,
            "use_block_pos_embedding": False,
            "use_pos_encoder": False,
            "use_block_signature": True,
            "use_ema": False,
            "steps": 10,
            "warmup_steps": 1,
            "test_steps": 1,
            "log_interval": 1,
        },
    )[0]
    job.seed = 0
    job.ablation_variant = "transfer_probe"
    job.output_dir = str(output_dir / "source")
    result = handler(job)
    task_dir = Path(result["run_dir"])
    with (task_dir / "metrics_steps.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert Counter(row["task_id"] for row in rows) == {
        "0": 11, "1": 10, "2": 10, "3": 10, "4": 10,
    }, "Source did not complete all five tasks."
    for row in rows:
        for name in ("loss", "meta_grad_norm", "weight_update_layer_mean"):
            assert math.isfinite(float(row[name])), f"Nonfinite source {name}."
    assert any(float(row["meta_grad_norm"]) > 0 for row in rows)
    state = torch.load(
        task_dir.parents[1] / "models/transformer_model.pth",
        map_location="cpu", weights_only=True,
    )
    assert all(torch.isfinite(value).all() for value in state.values())
    first_task = torch.load(
        task_dir / "model_checkpoint_task0_step11.pth",
        map_location="cpu", weights_only=True,
    )["transformer_state_dict"]
    assert not torch.equal(state["linear_blocks.weight"], first_task["linear_blocks.weight"]), (
        "The source update head did not change after the first task."
    )
    checkpoint = {
        "config": {**vars(job), "device": str(job.device)},
        "transformer_state": {
            name: value for name, value in state.items()
            if not name.startswith("te_model.")
        },
        "task_encoder_state": {
            name.removeprefix("te_model."): value for name, value in state.items()
            if name.startswith("te_model.")
        },
    }
    torch.save(checkpoint, output_dir / "shared_transformer.pth")


def validate_invalid_inputs(transformer, checkpoint, scope, batch):
    shared_state = checkpoint["transformer_state"]
    missing_state = dict(shared_state)
    del missing_state["linear_blocks.weight"]
    invalid_states = {
        "missing_key": missing_state,
        "extra_key": {**shared_state, "unexpected.weight": torch.zeros(1)},
        "wrong_shape": {**shared_state, "linear_blocks.weight": torch.zeros(1)},
    }
    for case, state in invalid_states.items():
        try:
            load_shared_weights(transformer, state)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted incompatible checkpoint: {case}")
    config = checkpoint["config"]
    invalid_transformer = TransformerModel(
        config, scope, task_encoder=LambdaLayer(
            lambda inputs: inputs.new_zeros(config["embedding_size"] + 1)
        ),
    )
    try:
        invalid_transformer(batch.x_sp, {}, {})
    except ValueError as error:
        assert "Task encoder output dim" in str(error), str(error)
    else:
        raise AssertionError("Accepted an incompatible encoder width.")


def run_target(model, encoder, task_data, checkpoint, output_dir, expected_tensors):
    output_dir.mkdir()
    config = checkpoint["config"]
    model = model.to(config["device"])
    model.pred_with_transformer = ["features"]
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
            nn.init.zeros_(module.bias)
    initial_parameters = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    scope = summarize_train_scope(
        model, include_tokens=model.pred_with_transformer,
        min_init_std=config["min_init_std"],
    )
    assert scope["selected_parameter_tensors"] == expected_tensors, (
        "Unexpected number of tensors assigned to the learned optimizer."
    )
    transformer = TransformerModel(config, scope, task_encoder=encoder)
    load_shared_weights(transformer, checkpoint["transformer_state"])
    transformer.requires_grad_(False)
    transformer.eval()
    frozen_state = {
        name: value.clone() for name, value in transformer.state_dict().items()
    }
    for name, value in checkpoint["transformer_state"].items():
        assert torch.equal(frozen_state[name], value), name
    trainer = PROPOSED(
        model, transformer, nn.CrossEntropyLoss(), config,
        batch_generator_class=dataset.BatchGenerator,
    )
    assert not trainer.theta["transformer"]
    managed_names = set(scope["selected_param_names"])
    constant_names = {
        name for name, value in initial_parameters.items()
        if value.std().item() < config["min_init_std"]
    }
    assert constant_names and constant_names.isdisjoint(managed_names)
    adam_parameters = {
        id(parameter) for parameter in trainer.model_optimizer.param_groups[0]["params"]
    }
    assert adam_parameters == {
        id(parameter) for name, parameter in model.named_parameters()
        if name not in managed_names
    }
    generator = dataset.BatchGenerator(task_data, config)
    with (output_dir / "metrics.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("step", "support_loss", "max_abs_update"))
        writer.writeheader()
        for step in range(10):
            batch = generator.get_batch()
            _, _, weights_info = trainer.get_updated_params(trainer.theta, batch, c=None)
            updates = weights_info["update"]
            assert updates.shape == (scope["selected_total_parameters"],)
            assert torch.isfinite(updates).all(), "Nonfinite learned update."
            # Importance calculation clears gradients, so Adam needs a fresh backward pass.
            trainer.model_optimizer.zero_grad(set_to_none=True)
            loss = trainer.criterion(model(batch.x_sp), batch.y_sp)
            assert torch.isfinite(loss), "Nonfinite target loss."
            loss.backward()
            trainer.model_optimizer.step()
            torch.testing.assert_close(transformer.state_dict(), frozen_state, rtol=0, atol=0)
            writer.writerow({
                "step": step + 1,
                "support_loss": loss.item(),
                "max_abs_update": updates.abs().max().item(),
            })
    changed_names = {
        name for name, value in model.named_parameters()
        if not torch.equal(value, initial_parameters[name])
    }
    assert managed_names <= changed_names, "Some learned-managed tensors did not change."
    assert constant_names <= changed_names, "Some constant tensors did not receive Adam updates."
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    assert not trainer.transformer_optimizer.state
    validate_invalid_inputs(transformer, checkpoint, scope, batch)
    torch.testing.assert_close(transformer.state_dict(), frozen_state, rtol=0, atol=0)
    print(f"PASS {type(task_data).__name__}: 10 updates, {expected_tensors} learned tensors")


def run_experiment(output_dir):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    train_source(output_dir)
    device, _ = set_device()
    checkpoint = torch.load(
        output_dir / "shared_transformer.pth", map_location=device, weights_only=True
    )
    _set_reproducibility(0)
    config = checkpoint["config"]
    mnist_encoder = _build_task_encoder(config, device)
    mnist_encoder.load_state_dict(checkpoint["task_encoder_state"], strict=True)
    mnist_tasks, _ = dataset.split_task_construction("splitmnist", [[0, 1]])
    cifar_tasks, _ = dataset.split_task_construction("cifar100", [[0, 1]])

    mnist_features = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 16),
        nn.Tanh(),
        nn.Linear(16, 8),
        nn.Tanh(),
        nn.LayerNorm(8),
    )
    mnist_model = nn.Sequential(OrderedDict([
        ("features", mnist_features),
        ("classifier", nn.Linear(8, 2)),
    ]))
    cifar_features = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.Tanh(),
        nn.Conv2d(4, 8, 3, padding=1),
        nn.Tanh(),
        nn.Conv2d(8, 8, 3, padding=1),
        nn.Tanh(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.LayerNorm(8),
    )
    cifar_model = nn.Sequential(OrderedDict([
        ("features", cifar_features),
        ("classifier", nn.Linear(8, 2)),
    ]))
    cifar_encoder = make_rgb_encoder(config["embedding_size"])
    run_target(
        mnist_model, mnist_encoder, mnist_tasks[0], checkpoint,
        output_dir / "mnist_mlp", expected_tensors=2,
    )
    run_target(
        cifar_model, cifar_encoder, cifar_tasks[0], checkpoint,
        output_dir / "cifar100_cnn", expected_tensors=3,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/transformer_transfer"))
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True)
    run_experiment(output_dir)


if __name__ == "__main__":
    main()
