from experiments.run_plans import PLANS, ProposedRun, build_proposed_jobs
from exp_runner_1_1_config import (
    ablation_jobs,
    proposed_primary_jobs,
    variant_rows,
)
from exp_runner_1_1_fn_setup import _experiment_label


def test_anchor_enables_features_without_changing_primary_defaults():
    anchor = ablation_jobs("splitmnist", "all_but_head", anchor_only=True)[0]
    feature_names = (
        "use_layer_id",
        "use_block_pos_embedding",
        "use_weight_stats",
        "use_block_signature",
        "use_stability",
        "use_tensor_embedding",
        "use_signed_gradient_basis",
        "use_pos_encoder",
        "use_ema",
    )
    assert all(getattr(anchor, name) is True for name in feature_names)
    assert anchor.ema_objective == "mas"
    assert anchor.plasticity_scales == [0.1, 0.33, 1.0]

    primary = proposed_primary_jobs(["splitmnist"], ["all_but_head"])[0]
    assert primary.use_pos_encoder is False
    assert primary.use_ema is False
    assert "use_block_pos_embedding" not in vars(primary)
    assert primary.use_layer_id is True


def test_ema_axis_contains_one_run_per_effective_setting():
    for overrides in ({}, {"ema_objective": "ce"}, {"use_ema": False}):
        jobs = ablation_jobs(
            "splitmnist",
            "all_but_head",
            feature_variants=variant_rows("anchor"),
            flag_overrides=overrides,
        )
        settings = {
            (job.use_ema, job.ema_objective if job.use_ema else None)
            for job in jobs
        }
        assert len(jobs) == 3
        assert settings == {(True, "mas"), (True, "ce"), (False, None)}
        assert len({_experiment_label(vars(job)) for job in jobs}) == 3


def test_embedding_ablations_disable_exactly_one_feature():
    disabled_features = {
        "no_layer_embedding": "use_layer_id",
        "no_block_position": "use_block_pos_embedding",
        "no_sequence_position": "use_pos_encoder",
    }
    plan = {
        "seeds": [0],
        "proposed": [ProposedRun(
            "splitmnist", "all_but_head", "all_but_head",
            variants=("anchor", *disabled_features),
        )],
    }
    anchor, *ablations = build_proposed_jobs(plan, "outputs")
    anchor_config = vars(anchor)
    assert len(ablations) == len(disabled_features)
    for job in ablations:
        disabled_feature = disabled_features[job.ablation_variant]
        assert vars(job) == {
            **anchor_config,
            "ablation_variant": job.ablation_variant,
            disabled_feature: False,
        }
        assert getattr(anchor, disabled_feature) is True
        assert job.ablation_variant in _experiment_label(vars(job))
    assert len({_experiment_label(vars(job)) for job in [anchor, *ablations]}) == 4


def test_grouped_identity_variants_keep_their_original_scope():
    no_layer, no_identity = ablation_jobs(
        "splitmnist",
        "all_but_head",
        include_ema_axis=False,
        feature_variants=variant_rows("no_layer_id", "no_identity"),
    )
    for job in (no_layer, no_identity):
        assert job.use_layer_id is False
        assert job.use_block_pos_embedding is False
    assert no_layer.use_tensor_embedding is True
    assert no_layer.use_block_signature is True
    assert no_identity.use_tensor_embedding is False
    assert no_identity.use_block_signature is False
    assert no_identity.use_pos_encoder is False


def test_existing_plans_keep_explicit_feature_choices():
    jobs = build_proposed_jobs(PLANS["single_no_ema"], "outputs")
    assert all(job.use_ema is False for job in jobs)
    assert all(job.use_tensor_embedding is False for job in jobs)

    jobs = build_proposed_jobs(PLANS["final_new_configs_50k"], "outputs")
    for job in jobs:
        assert job.use_ema is True and job.ema_objective == "mas"
        assert job.use_pos_encoder is False
        assert job.use_tensor_embedding is False
        if not job.use_layer_id:
            assert job.use_block_pos_embedding is False

    for run in PLANS["all_ablations_10k"]["proposed"]:
        assert "anchor" in run.variants and "no_ema" in run.variants
        assert "ema_mas" not in run.variants


def test_new_anchor_does_not_reuse_the_legacy_output_label():
    anchor = ablation_jobs("splitmnist", "all_but_head", anchor_only=True)[0]
    legacy = {**vars(anchor), "use_pos_encoder": False, "use_ema": False}
    assert _experiment_label(legacy) == (
        "row__br4_rs1.0_ps3_lid_ws_stab_anchor_tl-all_but_head"
    )
    configurations = (
        legacy,
        {**legacy, "use_pos_encoder": True},
        {**legacy, "use_ema": True},
        vars(anchor),
    )
    assert len({_experiment_label(config) for config in configurations}) == 4
