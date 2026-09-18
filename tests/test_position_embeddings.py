from itertools import product

import torch
from torch import nn

from block_level_learned_optimization.parameter_scope import summarize_train_scope
from block_level_learned_optimization.transformer import TransformerModel
from experiments.run_plans import ProposedRun, build_proposed_jobs
from exp_runner_1_1_fn_setup import _experiment_label


def build_model(flags):
    torch.manual_seed(7)
    base_model = nn.Sequential(
        nn.Linear(3, 4, bias=False), nn.Linear(4, 2, bias=False)
    )
    config = {
        "device": "cpu",
        "embedding_size": 8,
        "num_heads": 2,
        "dim_feedforward": 16,
        "n_layers": 1,
        "block_rows": 2,
        **flags,
    }
    model = TransformerModel(
        config, summarize_train_scope(base_model), task_encoder=nn.Identity()
    )
    return model, dict(base_model.named_parameters())


def forward_and_backward(model, weights):
    importance = {
        name: torch.full_like(weight, 0.5) for name, weight in weights.items()
    }
    outputs = model(torch.arange(8).float().reshape(1, 8) / 10, weights, importance)
    sum(output.square().sum() for output in outputs).backward()
    return outputs


def test_embeddings_can_be_enabled_independently():
    for use_layer_id, use_block_position in product((False, True), repeat=2):
        model, weights = build_model(
            {
                "use_layer_id": use_layer_id,
                "use_block_pos_embedding": use_block_position,
            }
        )
        outputs = forward_and_backward(model, weights)
        assert all(torch.isfinite(output).all() for output in outputs)
        assert hasattr(model, "layer_embedding") == use_layer_id
        assert hasattr(model, "block_pos_embedding") == use_block_position
        embeddings = [
            module for module in model.modules() if isinstance(module, nn.Embedding)
        ]
        for embedding in embeddings:
            gradient = embedding.weight.grad
            assert gradient is not None and torch.isfinite(gradient).all()
            assert torch.count_nonzero(gradient) > 0


def test_omitted_block_flag_preserves_legacy_behavior():
    for flags in ({}, {"use_layer_id": False}, {"use_layer_id": True}):
        legacy, legacy_weights = build_model(flags)
        explicit, explicit_weights = build_model(
            {
                **flags,
                "use_block_pos_embedding": flags.get("use_layer_id", True),
            }
        )
        torch.testing.assert_close(
            legacy.state_dict(), explicit.state_dict(), rtol=0, atol=0
        )
        explicit.load_state_dict(legacy.state_dict(), strict=True)
        torch.testing.assert_close(
            forward_and_backward(legacy, legacy_weights),
            forward_and_backward(explicit, explicit_weights),
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            {name: parameter.grad for name, parameter in legacy.named_parameters()},
            {name: parameter.grad for name, parameter in explicit.named_parameters()},
            rtol=0, atol=0,
        )


def test_run_labels_distinguish_independent_embedding_settings():
    labels = {}
    for use_layer_id, use_block_position in product((False, True), repeat=2):
        run = ProposedRun(
            "splitmnist", "all_but_head", "all_but_head",
            flag_overrides={
                "use_layer_id": use_layer_id,
                "use_block_pos_embedding": use_block_position,
            },
        )
        job = build_proposed_jobs({"seeds": [0], "proposed": [run]}, "outputs")[0]
        assert job.use_block_pos_embedding == use_block_position
        labels[use_layer_id, use_block_position] = _experiment_label(vars(job))
    assert len(set(labels.values())) == 4
    assert labels[True, True] == "row__br4_rs1.0_ps3_lid_ws_stab_tl-all_but_head"
    assert labels[False, False] == "row__br4_rs1.0_ps3_nolid_ws_stab_tl-all_but_head"
