"""Train two small CPU tasks and reload the same setup with EMA disabled."""

import argparse
from collections import namedtuple
from pathlib import Path

import torch
from torch import nn

from block_level_learned_optimization.parameter_scope import (
    DEFAULT_MIN_INIT_STD,
    summarize_train_scope,
)
from block_level_learned_optimization.task_encoder import TaskEncoderGeneric
from block_level_learned_optimization.trainer import PROPOSED
from block_level_learned_optimization.transformer import TransformerModel

Episode = namedtuple("Episode", ["x_sp", "y_sp", "x_qr", "y_qr"])


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(4, 8, bias=False)
        self.classifier = nn.Linear(8, 2, bias=False)
        self.pred_with_transformer = ["features.weight"]

    def forward(self, inputs):
        return self.classifier(torch.tanh(self.features(inputs)))


class EpisodeSampler:
    def __init__(self, data, config):
        self.inputs, self.targets = data
        self.device = config["device"]

    def get_batch(self, device=None):
        if device is None:
            device = self.device
        indices = torch.randperm(len(self.inputs))
        support_indices, query_indices = indices[:8], indices[8:24]
        return Episode(
            self.inputs[support_indices].to(device),
            self.targets[support_indices].to(device),
            self.inputs[query_indices].to(device),
            self.targets[query_indices].to(device),
        )


def build_trainer(config, initial_model_state=None):
    model = Classifier()
    if initial_model_state is not None:
        model.load_state_dict(initial_model_state)
    initial_model_state = {
        name: value.clone() for name, value in model.state_dict().items()
    }
    train_scope = summarize_train_scope(
        model,
        include_tokens=model.pred_with_transformer,
        min_init_std=config.get("min_init_std", DEFAULT_MIN_INIT_STD),
    )
    task_encoder = TaskEncoderGeneric(config)
    # Functional updates need initialized parameters, including LazyLinear.
    task_encoder(torch.zeros(1, 4))
    transformer_model = TransformerModel(
        config, train_scope, task_encoder=task_encoder
    )
    trainer = PROPOSED(
        model,
        transformer_model,
        nn.CrossEntropyLoss(),
        config,
        batch_generator_class=EpisodeSampler,
    )
    return trainer, initial_model_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/custom_data")
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True)
    checkpoint_path = arguments.output_dir / "checkpoint.pth"

    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = {
        "device": "cpu",
        "embedding_size": 8,
        "num_heads": 2,
        "dim_feedforward": 16,
        "n_layers": 1,
        "block_strategy": "row_wise",
        "block_rows": 2,
        "lr": 1e-3,
        "lr_transformer": 1e-3,
        "lr_inner": 1e-2,
        "lambda_l2": 1e-4,
        "top_K": 1.0,
        "inner_steps": 2,
        "test_steps": 2,
        "steps": 10,
        "warmup_steps": 1,
        "momentum": 0.0,
        "c": 0.8,
        "use_signed_gradient_basis": True,
        "plasticity_scales": [0.1, 1.0],
        "use_stability": True,
        "use_ema": False,
    }
    inputs = torch.randn(96, 4)
    tasks = [
        (inputs, (inputs[:, 0] + inputs[:, 1] > 0).long()),
        (inputs, (inputs[:, 2] - inputs[:, 3] > 0).long()),
    ]
    trainer, initial_model_state = build_trainer(config)
    initial_update_head = (
        trainer.transformer_model.linear_blocks.weight.detach().clone()
    )
    for task_id, data in enumerate(tasks):
        trainer.fit(data, task_id, print_output=False)
    assert not torch.equal(
        initial_update_head, trainer.transformer_model.linear_blocks.weight
    ), "Meta-training did not change the learned update head."

    # Initial weights preserve per-tensor update scales when rebuilding the trainer.
    torch.save(
        {
            "config": config,
            "initial_model_state": initial_model_state,
            "model_state_dict": trainer.model.state_dict(),
            "transformer_state_dict": trainer.transformer_model.state_dict(),
            "model_optimizer_state_dict": trainer.model_optimizer.state_dict(),
            "transformer_optimizer_state_dict": (
                trainer.transformer_optimizer.state_dict()
            ),
            "previous_task_score": trainer.previous_tsk_score,
            "task_id": trainer.task,
        },
        checkpoint_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored_trainer, _ = build_trainer(
        checkpoint["config"], checkpoint["initial_model_state"]
    )
    restored_trainer.model.load_state_dict(checkpoint["model_state_dict"])
    restored_trainer.transformer_model.load_state_dict(
        checkpoint["transformer_state_dict"]
    )
    restored_trainer.model_optimizer.load_state_dict(
        checkpoint["model_optimizer_state_dict"]
    )
    restored_trainer.transformer_optimizer.load_state_dict(
        checkpoint["transformer_optimizer_state_dict"]
    )
    restored_trainer.previous_tsk_score = checkpoint["previous_task_score"]
    restored_trainer.task = checkpoint["task_id"]

    with torch.no_grad():
        torch.testing.assert_close(
            trainer.model(inputs), restored_trainer.model(inputs), rtol=0, atol=0
        )
    # Two query measurements check the result after one support adaptation.
    episode = EpisodeSampler(tasks[-1], config).get_batch()
    original_result = trainer.adapt_and_evaluate(episode, task_id=len(tasks) - 1)
    restored_result = restored_trainer.adapt_and_evaluate(
        episode, task_id=len(tasks) - 1
    )
    assert original_result == restored_result
    print(f"Trained {len(tasks)} tasks and saved {checkpoint_path.resolve()}")
    print("Restored predictions and adaptation results match exactly.")


if __name__ == "__main__":
    main()
