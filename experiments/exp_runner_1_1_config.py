import warnings
from types import SimpleNamespace

DEFAULT_OUTPUT_DIR = "/projects/prjs2027/final_thesis_outputs/31_final_runs"
# DEFAULT_OUTPUT_DIR = "/projects/prjs2027/final_thesis_outputs/31_final_big_models_seed42"
# DEFAULT_OUTPUT_DIR = "/projects/prjs2027/final_thesis_outputs/31_imagenet_memory_usage"
# DEFAULT_OUTPUT_DIR = "/projects/prjs2027/test_blocks_final/31_check_blocksize_resnet_vit"


DEFAULT_STEPS = {
    "steps": 50000,
    "warmup_steps": 1500,
    "test_steps": 50,
}

STEPS_BY_DATASET = {}

# STEPS_BY_DATASET = {
#     # "splitmnist": {
#     #     "steps": 50000,
#     #     "warmup_steps": 1500,
#     #     "test_steps": 50,
#     # },
#     # "rotatedmnist": {
#     #     "steps": 50000,
#     #     "warmup_steps": 1500,
#     #     "test_steps": 50,
#     # },
#     # "cifar100": {
#     #     "steps": 50000,
#     #     "warmup_steps": 1500,
#     #     "test_steps": 50,
#     # },
#     # "cifar100_resnet": {
#     #     "steps": 2000,
#     #     "warmup_steps": 50,
#     #     "test_steps": 50,
#     # },
#     # "cifar100_vit": {
#     #     "steps": 50000,
#     #     "warmup_steps": 1500,
#     #     "test_steps": 50,
#     # },
#     # "tinyimagenet_resnet": {
#     #     "steps": 50000,
#     #     "warmup_steps": 1500,
#     #     "test_steps": 50,
#     # },
#     # "tinyimagenet_vit": {
#     #     "steps": 50000,
#     #     "warmup_steps": 1500,
#     #     "test_steps": 50,
#     # },
# }


def steps_for(dataset):
    """Per-dataset step budget; missing keys fall back to DEFAULT_STEPS."""
    return {**DEFAULT_STEPS, **STEPS_BY_DATASET.get(dataset, {})}


FEATURE_FLAGS_DEFAULT = {
    "use_layer_id": True,
    "use_weight_stats": True,
    "use_block_signature": True,
    "use_stability": True,
    "use_tensor_embedding": True,
    "use_signed_gradient_basis": True,
    "use_pos_encoder": False,
    "use_ema": False,
    "ema_objective": "mas",
}

DATASET_FEATURE_OVERRIDES = {
    "splitmnist": {},
    "rotatedmnist": {},
    "cifar100": {},
    "cifar100_resnet": {},
    "cifar100_vit": {},
    "tinyimagenet_resnet": {},
    "tinyimagenet_vit": {},
}


def feature_flags_for(dataset):
    """Effective proposer feature flags for one dataset."""
    return {
        **FEATURE_FLAGS_DEFAULT,
        **DATASET_FEATURE_OVERRIDES.get(dataset, {}),
    }


DATASET_CONFIGS = {
    "splitmnist": {
        "base_dataset": "splitmnist",
        "model_class": "ConvNet",
        "input_dim": 1,
        "input_size": 28,
        "num_labels": 10,
        "n": 2,
        "features_dim": 32,
        "inner_steps": 3,
        "lr": 1e-3,  # Adam for classifier head
        "lr_inner": 1e-2,
        "lr_inner_test": 1e-3,
        "lambda_l2": 1e-4,
        "clamping_transformer": 1.0,
        "spatial": None,
        "embedding_size": 16,
    },
    "rotatedmnist": {
        "base_dataset": "rotatedmnist",
        "model_class": "ConvNet",
        "input_dim": 1,
        "input_size": 28,
        "num_labels": 10,
        "n": 10,
        "features_dim": 64,
        "inner_steps": 3,
        "lr": 1e-4,  # Adam for classifier head
        "lr_inner": 1e-3,
        "lr_inner_test": 1e-4,
        "lambda_l2": 1e-5,
        "clamping_transformer": 2.0,  # prep-phase value
        "spatial": None,
        "embedding_size": 16,
    },
    "cifar100": {
        "base_dataset": "cifar100",
        "model_class": "ThreeConvNetSimple",
        "input_dim": 3,
        "input_size": 32,
        "num_labels": 100,
        "n": 5,
        "features_dim": 100,
        "inner_steps": 5,
        "lr": 1e-4,  # Adam for classifier head
        "lr_inner": 1e-3,
        "lr_inner_test": 1e-5,
        "lambda_l2": 1e-4,
        "clamping_transformer": 3.0,
        "spatial": 10000,
        "embedding_size": 16,
        "lr_transformer": 1e-4,  # transformer meta-optimizer
    },
    "cifar100_resnet": {
        "base_dataset": "cifar100",
        "model_class": "ResNet18",
        "input_dim": 3,
        "input_size": 32,
        "num_labels": 100,
        "n": 5,
        "features_dim": 512,
        "inner_steps": 5,
        "lr": 1e-4,  # Adam for classifier head
        "lr_inner": 1e-3,
        "lr_inner_test": 1e-5,
        "lambda_l2": 1e-4,
        "clamping_transformer": 2.0,
        "spatial": 10000,
        "embedding_size": 16,
        "lr_transformer": 1e-4,  # transformer meta-optimizer
    },
    "cifar100_vit": {
        "base_dataset": "cifar100",
        "model_class": "ViTSmall",
        "input_dim": 3,
        "input_size": 32,
        "num_labels": 100,
        "n": 5,
        "features_dim": 192,
        "inner_steps": 5,
        "lr": 1e-4,  # Adam for classifier head
        "lr_inner": 1e-3,
        "lr_inner_test": 1e-5,
        "lambda_l2": 1e-4,
        "clamping_transformer": 2.0,
        "spatial": None,
        "vit_patch_size": 4,
        "vit_num_heads": 6,
        "vit_num_layers": 6,
        "embedding_size": 16,
        "lr_transformer": 1e-4,  # transformer meta-optimizer
    },
    "tinyimagenet_resnet": {
        "base_dataset": "tinyimagenet",
        "model_class": "ResNet18",
        "input_dim": 3,
        "input_size": 64,
        "num_labels": 200,
        "n": 20,  # 10way head
        "k": 20,  # 10 support images
        "features_dim": 512,
        "inner_steps": 5,
        "lr": 1e-4,
        "lr_inner": 1e-3,
        "lr_inner_test": 1e-5,
        "lambda_l2": 5e-7,
        "clamping_transformer": 2.0,
        "spatial": 10000,
        "embedding_size": 16,
        "lr_transformer": 1e-4,  # transformer meta-optimizer
    },
    "tinyimagenet_vit": {
        "base_dataset": "tinyimagenet",
        "model_class": "ViTSmall",
        "input_dim": 3,
        "input_size": 64,
        "num_labels": 200,
        "n": 20,  # 10way head
        "k": 20,  # 10 support images
        "features_dim": 192,
        "inner_steps": 5,
        "lr": 1e-4,
        "lr_inner": 1e-3,
        "lr_inner_test": 1e-5,
        "lambda_l2": 5e-7,
        "clamping_transformer": 2.0,
        "spatial": None,
        "vit_patch_size": 8,
        "vit_num_heads": 6,
        "vit_num_layers": 6,
        "embedding_size": 16,
        "lr_transformer": 1e-4,  # transformer meta-optimizer
    },
}


BASE_CONFIG = {
    "seed": 0,
    "method": "proposed",
    "multihead": False,
    "batch_size": 32,
    "k": 5,
    "dropout": 0.2,
    "task_encoder": True,
    "num_heads": 4,
    "dim_feedforward": 64,
    "n_layers": 2,
    "use_bn": True,
    "predict_bn": False,
    "min_init_std": 1e-6,
    "unique_te": True,
    "top_K": 0.4,
    "importance_eps": 1e-3,
    "momentum": 0.99,
    "c": 0.8,
    "lr_transformer": 1e-3,
    "lr_inner_transformer": 1e-3,
    "tensor_embed_dim": 8,
    "ema_beta": 0.99,
    "log_interval": 500,
    "checkpoint_interval": 5000,
    "outer_optimizer": "adam",
    "bootstrap_resamples": 1000,
    "bootstrap_confidence": 0.95,
}


BLOCK_ROWS_BY_DATASET = {
    "splitmnist": 4,  # 32 blocks at 1, one per filter, 4 did better
    "rotatedmnist": 4,  # best empirically+prep phase
    "cifar100": 1,  # 300 blocks across conv1-3, to try 5 n 10, divides 100 well, at 10 macro view.
    "cifar100_resnet": 8,  # 2,560 blocks, one per layer4 filter at 1. 64 blocks at 8 and 32 blocks at 16 for a 512-channel layer, 32 macro view, 8 granular
    "tinyimagenet_resnet": 8,  # 2,560 blocks, one per layer4 filter at 1. 64 blocks at 8 and 32 blocks at 16 for a 512-channel layer, 32 macro view, 8 granular
    "cifar100_vit": 32,  # 32 does 1 head, 96 groups 3 heads, divides 192,576 768 well # head_dim, at 32, 54 head-aligned blocks in the last layer
    "tinyimagenet_vit": 32,  # 32 does 1 head, 96 groups 3 heads, divides 192,576 768 well
}
DEFAULT_BLOCK_ROWS = 1


def block_rows_for(dataset):
    return BLOCK_ROWS_BY_DATASET.get(dataset, DEFAULT_BLOCK_ROWS)


def normalize_training_layers(training_layers):
    """Canonicalizes a training_layers spec.

    None and "all" mean every parameter-owning module is transformer-managed.
    A list means exactly those modules; everything else is Adam-managed.
    """
    if training_layers is None:
        return None
    if isinstance(training_layers, str):
        key = training_layers.lower()
        if key == "all":
            return "all"
        if key == "all_but_head":
            return "all_but_head"
        raise ValueError(
            f"training_layers string must be 'all' or 'all_but_head'. "
            f"Got '{training_layers}'."
        )
    if isinstance(training_layers, (list, tuple)):
        if len(training_layers) == 0:
            raise ValueError(
                "training_layers list is empty. Use None or 'all' for all layers."
            )
        return list(training_layers)
    raise TypeError(
        f"training_layers must be None, 'all', or a list. Got {type(training_layers).__name__}."
    )


def make_experiment_config(dataset, extra, training_layers=None):
    """Builds one experiment config by overlaying, in order: base, the
    dataset's feature flags, the dataset config, the dataset's step budget,
    then the job-specific extras."""
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Known: {sorted(DATASET_CONFIGS.keys())}."
        )

    merged = {
        **BASE_CONFIG,
        **feature_flags_for(dataset),
        **DATASET_CONFIGS[dataset],
        **steps_for(dataset),
        **extra,
        "dataset": dataset,
        "training_layers": normalize_training_layers(training_layers),
    }
    return SimpleNamespace(**merged)


PLASTICITY_SCALES_DEFAULT = [
    [0.1, 0.33, 1.0],
]


def proposed_primary_jobs(
    datasets,
    training_layers_variants=None,
    plasticity_scales_list=None,
    flag_overrides=None,
):
    """Primary proposed-method sweep.

    Feature flags default to feature_flags_for(dataset); explicit overrides
    take precedence over dataset and block settings.
    """
    jobs = []
    if training_layers_variants is None:
        training_layers_variants = [None]
    if plasticity_scales_list is None:
        plasticity_scales_list = PLASTICITY_SCALES_DEFAULT

    for dataset_name in datasets:
        row_wise_block_configs = [
            {"block_rows": block_rows_for(dataset_name), "row_scale": 1.0},
        ]
        for training_layers in training_layers_variants:
            for plasticity_scales in plasticity_scales_list:
                for block_config in row_wise_block_configs:
                    jobs.append(
                        make_experiment_config(
                            dataset_name,
                            {
                                "block_strategy": "row_wise",
                                "plasticity_scales": plasticity_scales,
                                "alpha": None,
                                "beta": None,
                                **block_config,
                                **(flag_overrides or {}),
                            },
                            training_layers=training_layers,
                        )
                    )
    return jobs


def proposed_ablation_jobs(
    datasets,
    training_layers_variants=None,
    best_plasticity_scales=None,
    best_block_rows=4,
):
    jobs = []
    if training_layers_variants is None:
        training_layers_variants = [None]
    if best_plasticity_scales is None:
        best_plasticity_scales = [0.1, 0.4, 1.0]

    base_block = {
        "block_strategy": "row_wise",
        "block_rows": best_block_rows,
        "row_scale": 1.0,
        "alpha": None,
        "beta": None,
    }
    base_flags = {
        "use_layer_id": True,
        "use_weight_stats": True,
        "use_stability": True,
    }

    ablation_variants = [
        ("no_plasticity", {"plasticity_scales": None}),
        ("no_layer_id", {"use_layer_id": False}),
        ("no_weight_stats", {"use_weight_stats": False}),
        ("no_stability", {"use_stability": False}),
    ]

    for dataset_name in datasets:
        for training_layers in training_layers_variants:
            for variant_name, overrides in ablation_variants:
                extra = {
                    **base_block,
                    "plasticity_scales": best_plasticity_scales,
                    **base_flags,
                    **overrides,
                    "ablation_variant": variant_name,
                }
                jobs.append(
                    make_experiment_config(
                        dataset_name, extra, training_layers=training_layers
                    )
                )
    return jobs


ANCHOR_FLAGS = {
    **FEATURE_FLAGS_DEFAULT,
    "use_block_pos_embedding": True,
    "use_pos_encoder": True,
    "use_ema": True,
}


VARIANTS = {
    "anchor": {},
    "no_block_statistics": {
        "use_weight_stats": False,
        "use_block_signature": False,
    },
    "no_layer_id": {
        "use_layer_id": False,
        "use_block_pos_embedding": False,
    },
    "no_tensor_embedding": {"use_tensor_embedding": False},
    "no_weight_stats": {"use_weight_stats": False},
    "no_block_signature": {"use_block_signature": False},
    "no_identity": {
        "use_layer_id": False,
        "use_block_pos_embedding": False,
        "use_tensor_embedding": False,
        "use_block_signature": False,
        "use_pos_encoder": False,
    },
    "no_ema": {"use_ema": False},
    "ema_mas": {"use_ema": True, "ema_objective": "mas"},
    "ema_ce": {"use_ema": True, "ema_objective": "ce"},
}


def variant_rows(*names):
    """(name, overrides) pairs for the given variant names, in order."""
    return [(name, VARIANTS[name]) for name in names]


ABLATION_VARIANTS = variant_rows(
    "anchor",
    "no_block_statistics",
    "no_layer_id",
    "no_tensor_embedding",
)
GROUPED_VARIANTS = variant_rows("anchor", "no_block_statistics", "no_identity")
EMA_AXIS = variant_rows("no_ema", "ema_mas", "ema_ce")


def ablation_jobs(
    dataset,
    training_layers,
    anchor_only=False,
    include_method_justification=False,
    include_ema_axis=True,
    flag_overrides=None,
    drop_variants=None,
    feature_variants=None,
    plasticity_scales=None,
    block_rows=None,
    row_scale=1.0,
):
    """Builds the ablation matrix for one (dataset, training_layers) pair."""
    if include_method_justification:
        warnings.warn(
            "include_method_justification is deprecated and ignored: the "
            "signed-gradient basis is permanent, so no_signed_gradient_basis "
            "is no longer a buildable row.",
            DeprecationWarning,
            stacklevel=2,
        )
    flag_overrides = dict(flag_overrides or {})
    drop = set(drop_variants or ())
    anchor_flags = {**ANCHOR_FLAGS, **flag_overrides}
    base_variants = (
        feature_variants if feature_variants is not None else ABLATION_VARIANTS
    )

    if plasticity_scales is None:
        plasticity_scales = PLASTICITY_SCALES_DEFAULT[0]
    if block_rows is None:
        block_rows = block_rows_for(dataset)

    base_block = {
        "block_strategy": "row_wise",
        "block_rows": block_rows,
        "row_scale": row_scale,
        "alpha": None,
        "beta": None,
        "plasticity_scales": plasticity_scales,
    }

    if anchor_only:
        variants = list(base_variants[:1])
    else:
        variants = list(base_variants)
        if include_ema_axis:
            variants += [
                (name, overrides)
                for (name, overrides) in EMA_AXIS
                if any(
                    anchor_flags.get(flag) != value
                    for flag, value in overrides.items()
                )
            ]

    variants = [
        (name, overrides) for (name, overrides) in variants if name not in drop
    ]

    jobs = []
    for variant_name, overrides in variants:
        extra = {
            **base_block,
            **anchor_flags,
            **overrides,
            "ablation_variant": variant_name,
        }
        jobs.append(
            make_experiment_config(
                dataset, extra, training_layers=training_layers
            )
        )

    return jobs


def get_experiment_configs(
    datasets=None,
    training_layers_variants=None,
    plasticity_scales_list=None,
    include_ablations=False,
):
    if datasets is None:
        datasets = ["splitmnist", "rotatedmnist", "cifar100"]
    jobs = proposed_primary_jobs(
        datasets,
        training_layers_variants,
        plasticity_scales_list=plasticity_scales_list,
    )
    if include_ablations:
        jobs += proposed_ablation_jobs(datasets, training_layers_variants)
    return jobs
