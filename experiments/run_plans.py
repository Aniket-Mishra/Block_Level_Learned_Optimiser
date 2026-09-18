"""Plan-based multi-seed runner for Snellius. This is the single entry point."""

import argparse
import copy
import glob
import importlib
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset
import pandas as pd
import torch
from exp_runner_1_1_config import (
    DEFAULT_OUTPUT_DIR,
    VARIANTS,
    ablation_jobs,
    proposed_primary_jobs,
)
from exp_runner_1_1_fn_setup import _experiment_label

SEEDS = [0, 42, 67]


ALL = None
ALL_BUT_HEAD = "all_but_head"
CONV3 = ["conv3"]
RESNET_LAYER4_CONV = [
    "layer4.0.conv1",
    "layer4.0.conv2",
    "layer4.0.downsample.0",
    "layer4.1.conv1",
    "layer4.1.conv2",
]
VIT_LAST_LAYER_COMPUTE = [
    "encoder.layers.5.self_attn",
    "encoder.layers.5.linear1",
    "encoder.layers.5.linear2",
]


HANDLER_MAP = {
    "proposed": "exp_runner_1_1_fn_setup",
    "proposed_ablation": "exp_runner_1_1_fn_setup",
    "ewc_online": "baselines.ewc_online_handler",
    "ewc_vanilla": "baselines.EWC_Vanilla",
    "der_pp": "baselines.der_pp_handler",
    "er_ace": "baselines.er_ace_handler",
    "l2p": "baselines.l2p_handler",
    "dytox": "baselines.dytox_handler",
    "wsn": "baselines.wsn_handler",
    "lamaml": "baselines.lamaml_handler",
    "l2l": "baselines.l2l_handler",
}


def get_handler_fn(method_name):
    if method_name not in HANDLER_MAP:
        raise KeyError(
            f"Unknown method '{method_name}'. Known: {sorted(HANDLER_MAP.keys())}."
        )
    module = importlib.import_module(HANDLER_MAP[method_name])
    return module.handler


def run_single_job(args):
    """Runs one job in a worker process. Returns the result row, or a failure
    row (so a crash in one job never loses the rest of the batch)."""
    torch.set_num_threads(1)
    try:
        started = time.time()
        handler_fn = get_handler_fn(args.method)
        result = handler_fn(args)
        elapsed = time.time() - started
        exp_label = (
            result.get("exp_label") if isinstance(result, dict) else None
        )
        print(
            f"DONE: {args.dataset} {args.method} {exp_label or ''} ({elapsed:.1f}s)"
        )
        return result
    except Exception as err:
        import traceback

        traceback.print_exc()
        print(f"FAILED: {args.dataset} {args.method}: {err}")
        return {
            "dataset": getattr(args, "dataset", "unknown"),
            "method": getattr(args, "method", "unknown"),
            "error": str(err),
            "status": "failed",
        }


@dataclass
class ProposedRun:
    dataset: str
    training_layers: object
    layer_set_name: str
    variants: tuple = None
    flag_overrides: dict = field(default_factory=dict)
    block_rows: int = None


@dataclass
class BaselineRun:
    """One baseline comparison: every method in `methods` at this scope."""

    dataset: str
    training_layers: object
    layer_set_name: str
    methods: tuple


MNIST_BASELINES = ("ewc_vanilla", "der_pp", "er_ace", "wsn", "lamaml", "l2l")
CONV_BASELINES = ("ewc_vanilla", "der_pp", "er_ace", "wsn", "lamaml")
VIT_BASELINES = CONV_BASELINES + ("l2p", "dytox")
# imagenet_baseline_backfill = (
#     "ewc_vanilla",
#     "er_ace",
#     "wsn",
#     "der_pp",
#     "dytox",
# )
imagenet_baseline_backfill = ("lamaml",)


PLANS = {
    "splitmnist_smoke": {
        "seeds": [0],
        "proposed": [
            ProposedRun(
                "splitmnist",
                ALL_BUT_HEAD,
                "all_but_head",
                flag_overrides={
                    "steps": 10,
                    "warmup_steps": 1,
                    "test_steps": 1,
                    "log_interval": 1,
                },
            ),
        ],
        "baselines": [],
    },
    "mnist": {
        "proposed": [
            ProposedRun("splitmnist", ALL_BUT_HEAD, "all_but_head"),
            ProposedRun("rotatedmnist", ALL_BUT_HEAD, "all_but_head"),
        ],
        "baselines": [
            # BaselineRun("splitmnist", ALL_BUT_HEAD, "all_but_head", MNIST_BASELINES),
            # BaselineRun("rotatedmnist", ALL_BUT_HEAD, "all_but_head", MNIST_BASELINES),
        ],
    },
    "cifar": {
        "proposed": [
            ProposedRun("cifar100", CONV3, "conv3"),
            ProposedRun("cifar100", ALL_BUT_HEAD, "all_but_head"),
        ],
        "baselines": [
            BaselineRun(
                "cifar100", ALL_BUT_HEAD, "all_but_head", CONV_BASELINES
            ),
            BaselineRun("cifar100", CONV3, "conv3", ("l2l",)),
        ],
    },
    "cifar_big": {
        "proposed": [
            ProposedRun(
                "cifar100_resnet", RESNET_LAYER4_CONV, "RESNET_LAYER4_CONV"
            ),
            # ProposedRun("cifar100_vit", VIT_LAST_LAYER_COMPUTE, "VIT_LAST_LAYER_COMPUTE"),
        ],
        "baselines": [
            # BaselineRun("cifar100_resnet", ALL, "all", CONV_BASELINES),
            # BaselineRun("cifar100_vit", ALL, "all", VIT_BASELINES),
        ],
    },
    "tinyimagenet": {
        "proposed": [
            ProposedRun(
                "tinyimagenet_resnet", RESNET_LAYER4_CONV, "RESNET_LAYER4_CONV"
            ),
            # ProposedRun("tinyimagenet_vit", VIT_LAST_LAYER_COMPUTE, "VIT_LAST_LAYER_COMPUTE"),
        ],
        "baselines": [
            # BaselineRun("tinyimagenet_resnet", ALL, "all", CONV_BASELINES),
            # BaselineRun("tinyimagenet_vit", ALL, "all", VIT_BASELINES),
        ],
    },
    "l2l_baseline": {
        "proposed": [],
        "baselines": [
            BaselineRun("splitmnist", ALL_BUT_HEAD, "all_but_head", ("l2l",)),
            BaselineRun(
                "rotatedmnist", ALL_BUT_HEAD, "all_but_head", ("l2l",)
            ),
            BaselineRun("cifar100", CONV3, "conv3", ("l2l",)),
        ],
    },
    "single_no_ema": {
        "proposed": [
            ProposedRun(
                "splitmnist",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=("anchor",),
                flag_overrides={"use_tensor_embedding": False, "use_ema": False},
            ),
            ProposedRun(
                "rotatedmnist",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=("anchor",),
                flag_overrides={"use_tensor_embedding": False, "use_ema": False},
            ),
            ProposedRun(
                "cifar100",
                CONV3,
                "conv3",
                variants=("anchor",),
                flag_overrides={"use_tensor_embedding": False, "use_ema": False},
            ),
        ],
        "baselines": [],
    },
    "cifar_3conv_ablation": {
        "proposed": [
            ProposedRun(
                "cifar100",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=("anchor", "no_identity", "no_block_statistics"),
                block_rows=1,
            ),
        ],
        "baselines": [
            BaselineRun(
                "cifar100", ALL_BUT_HEAD, "all_but_head", CONV_BASELINES
            ),
        ],
    },
    "feature_ablation": {
        "proposed": [
            ProposedRun(
                "splitmnist",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=("no_layer_id", "no_block_statistics"),
                flag_overrides={"use_tensor_embedding": False},
            ),
            ProposedRun(
                "rotatedmnist",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=("no_layer_id", "no_block_statistics"),
                flag_overrides={"use_tensor_embedding": False},
            ),
            ProposedRun(
                "cifar100",
                CONV3,
                "conv3",
                variants=("no_layer_id", "no_block_statistics"),
                flag_overrides={"use_tensor_embedding": False},
            ),
            ProposedRun(
                "cifar100",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=("no_layer_id", "no_block_statistics"),
            ),
        ],
        "baselines": [],
    },
    "resnet": {
        "seeds": [42],
        "proposed": [
            # ProposedRun("cifar100_resnet", RESNET_LAYER4_CONV, "RESNET_LAYER4_CONV"),
            ProposedRun(
                "tinyimagenet_resnet", RESNET_LAYER4_CONV, "RESNET_LAYER4_CONV"
            ),
        ],
        "baselines": [
            # BaselineRun("cifar100_resnet", ALL, "all", CONV_BASELINES),
            # BaselineRun("tinyimagenet_resnet", ALL, "all", CONV_BASELINES),
        ],
    },
    "vit": {
        "seeds": [42, 67],
        "proposed": [
            # ProposedRun("cifar100_vit", VIT_LAST_LAYER_COMPUTE, "VIT_LAST_LAYER_COMPUTE"),
            # ProposedRun("tinyimagenet_vit", VIT_LAST_LAYER_COMPUTE, "VIT_LAST_LAYER_COMPUTE"),
        ],
        "baselines": [
            # Whole-model baselines (ALL); see the scope rule above.
            BaselineRun("cifar100_vit", ALL, "all", VIT_BASELINES),
            # BaselineRun("tinyimagenet_vit", ALL, "all", VIT_BASELINES),
        ],
    },
    "imagenet": {
        "seeds": [0],
        "proposed": [
            # ProposedRun("tinyimagenet_resnet", RESNET_LAYER4_CONV, "RESNET_LAYER4_CONV",
            #             variants=("anchor",)),
            # ProposedRun("tinyimagenet_vit", VIT_LAST_LAYER_COMPUTE, "VIT_LAST_LAYER_COMPUTE",
            #             variants=("anchor",)),
        ],
        "baselines": [
            # BaselineRun("tinyimagenet_resnet", ALL, "all", CONV_BASELINES),
            BaselineRun(
                "tinyimagenet_vit", ALL, "all", imagenet_baseline_backfill
            ),
        ],
    },
    "cifar_allbuthead_ablations_10k": {
        "seeds": [0],
        "proposed": [
            ProposedRun(
                "cifar100",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=(
                    "anchor",
                    "no_block_statistics",
                    "no_layer_id",
                    "no_tensor_embedding",
                ),
                flag_overrides={"steps": 10000, "warmup_steps": 300},
                block_rows=1,
            ),
            ProposedRun(
                "cifar100",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=(
                    "anchor",
                    "no_block_statistics",
                    "no_layer_id",
                    "no_tensor_embedding",
                ),
                flag_overrides={"steps": 10000, "warmup_steps": 300},
                block_rows=4,
            ),
        ],
        "baselines": [],
    },
    "splitmnist_ablations_10k": {
        "seeds": [0],
        "proposed": [
            ProposedRun(
                "splitmnist",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=(
                    "anchor",
                    "no_block_statistics",
                    "no_layer_id",
                    "no_tensor_embedding",
                ),
                flag_overrides={"steps": 10000, "warmup_steps": 300},
                block_rows=1,
            ),
            ProposedRun(
                "splitmnist",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=(
                    "anchor",
                    "no_block_statistics",
                    "no_layer_id",
                    "no_tensor_embedding",
                ),
                flag_overrides={"steps": 10000, "warmup_steps": 300},
                block_rows=4,
            ),
        ],
        "baselines": [],
    },
    "all_ablations_10k": {
        "seeds": [42],
        "proposed": [
            ProposedRun(
                "cifar100",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=(
                    "anchor",
                    "no_ema",
                    "no_block_statistics",
                    "no_layer_id",
                    "no_tensor_embedding",
                ),
                flag_overrides={"steps": 10000, "warmup_steps": 300},
                block_rows=1,
            ),
            ProposedRun(
                "splitmnist",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=(
                    "anchor",
                    "no_ema",
                    "no_block_statistics",
                    "no_layer_id",
                    "no_tensor_embedding",
                ),
                flag_overrides={"steps": 10000, "warmup_steps": 300},
                block_rows=4,
            ),
            ProposedRun(
                "splitmnist",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=(
                    "anchor",
                    "no_ema",
                    "no_block_statistics",
                    "no_layer_id",
                    "no_tensor_embedding",
                ),
                flag_overrides={"steps": 10000, "warmup_steps": 300},
                block_rows=1,
            ),
            ProposedRun(
                "cifar100",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=(
                    "anchor",
                    "no_ema",
                    "no_block_statistics",
                    "no_layer_id",
                    "no_tensor_embedding",
                ),
                flag_overrides={"steps": 10000, "warmup_steps": 300},
                block_rows=4,
            ),
        ],
        "baselines": [],
    },
    "block_rows_probe": {
        "seeds": [42],
        "proposed": [
            ProposedRun(
                "cifar100_resnet",
                RESNET_LAYER4_CONV,
                "RESNET_LAYER4_CONV",
                variants=("anchor",),
                flag_overrides={
                    "steps": 1000,
                    "warmup_steps": 50,
                    "test_steps": 10,
                    "max_tasks": 1,
                },
                block_rows=8,
            ),
            ProposedRun(
                "tinyimagenet_vit",
                VIT_LAST_LAYER_COMPUTE,
                "VIT_LAST_LAYER_COMPUTE",
                variants=("anchor",),
                flag_overrides={
                    "steps": 1000,
                    "warmup_steps": 50,
                    "test_steps": 10,
                    "max_tasks": 1,
                },
                block_rows=32,
            ),
        ],
        "baselines": [],
    },
    "usage_probe_vit": {
        "seeds": [0],
        "proposed": [
            ProposedRun(
                "tinyimagenet_vit",
                VIT_LAST_LAYER_COMPUTE,
                "VIT_LAST_LAYER_COMPUTE",
                variants=("anchor",),
                flag_overrides={
                    "steps": 200,
                    "warmup_steps": 0,
                    "test_steps": 10,
                },
                block_rows=32,
            ),
        ],
        "baselines": [],
    },
    "rotatedmnist_ema_seed67": {
        "seeds": [67],
        "proposed": [
            ProposedRun(
                "rotatedmnist",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=("ema_mas",),
                flag_overrides={"use_tensor_embedding": False},
                block_rows=4,
            ),
        ],
        "baselines": [],
    },
    "cifar_l2l_backfill": {
        "seeds": [0, 42, 67],
        "proposed": [],
        "baselines": [
            BaselineRun("cifar100", CONV3, "conv3", ("l2l",)),
        ],
    },
    "final_new_configs_50k": {
        "seeds": [0, 42, 67],
        "proposed": [
            ProposedRun(
                "splitmnist",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=("anchor",),
                flag_overrides={
                    "use_layer_id": False,
                    "use_block_pos_embedding": False,
                    "use_tensor_embedding": False,
                    "use_pos_encoder": False,
                    "use_ema": True,
                    "ema_objective": "mas",
                },
                block_rows=1,
            ),
            # RotatedMNIST: Block+EMA (drop LID), br4. vs LID+Block+EMA br4 (sum 68.8).
            ProposedRun(
                "rotatedmnist",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=("anchor",),
                flag_overrides={
                    "use_layer_id": False,
                    "use_block_pos_embedding": False,
                    "use_tensor_embedding": False,
                    "use_pos_encoder": False,
                    "use_ema": True,
                    "ema_objective": "mas",
                },
                block_rows=4,
            ),
            # CIFAR conv3: LID+Block+EMA (add EMA, keep LID), br1. vs LID+Block br1 (sum 37.9).
            ProposedRun(
                "cifar100",
                CONV3,
                "conv3",
                variants=("anchor",),
                flag_overrides={
                    "use_tensor_embedding": False,
                    "use_pos_encoder": False,
                    "use_ema": True,
                    "ema_objective": "mas",
                },
                block_rows=1,
            ),
            # CIFAR conv1-3: Block+EMA (drop LID, add EMA), br1. vs Block br1 (sum 40.2, the tradeoff-best).
            ProposedRun(
                "cifar100",
                ALL_BUT_HEAD,
                "all_but_head",
                variants=("anchor",),
                flag_overrides={
                    "use_layer_id": False,
                    "use_block_pos_embedding": False,
                    "use_tensor_embedding": False,
                    "use_pos_encoder": False,
                    "use_ema": True,
                    "ema_objective": "mas",
                },
                block_rows=1,
            ),
        ],
        "baselines": [],
    },
}


def build_proposed_jobs(plan, output_dir):
    jobs = []
    seeds = plan.get("seeds", SEEDS)
    for run in plan["proposed"]:
        if run.variants is None:
            base_jobs = proposed_primary_jobs(
                [run.dataset],
                [run.training_layers],
                flag_overrides=run.flag_overrides,
            )
        else:
            unknown = [name for name in run.variants if name not in VARIANTS]
            if unknown:
                raise ValueError(
                    f"Unknown ablation variants for {run.dataset}: {unknown}"
                )
            rows = [(name, VARIANTS[name]) for name in run.variants]
            base_jobs = ablation_jobs(
                run.dataset,
                run.training_layers,
                anchor_only=False,
                include_method_justification=False,
                include_ema_axis=False,
                feature_variants=rows,
                flag_overrides=run.flag_overrides,
            )
        for base_job in base_jobs:
            base_job.training_layers_name = run.layer_set_name
            if run.block_rows is not None:
                base_job.block_rows = run.block_rows
            for seed in seeds:
                job = copy.deepcopy(base_job)
                job.seed = seed
                job.output_dir = output_dir
                jobs.append(job)
    return jobs


def build_baseline_jobs(plan, output_dir):
    from baselines.baseline_config import make_baseline_config

    jobs = []
    seeds = plan.get("seeds", SEEDS)
    for run in plan["baselines"]:
        for method_name in run.methods:
            base_job = make_baseline_config(
                run.dataset, method_name, run.training_layers
            )
            if base_job is None:
                continue
            base_job.training_layers_name = run.layer_set_name
            for seed in seeds:
                job = copy.deepcopy(base_job)
                job.seed = seed
                job.output_dir = output_dir
                jobs.append(job)
    return jobs


def training_layers_label(training_layers):
    if training_layers is None:
        return "all"
    if isinstance(training_layers, str):
        return training_layers
    return ",".join(training_layers)


def load_finished_baseline_keys(output_dir):
    """(method, dataset, training_layers label, seed) for every non-failed
    baseline row already written to any chunk or combined CSV."""
    keys = set()
    pattern_list = [
        os.path.join(output_dir, "chunks", "chunk_*.csv"),
        os.path.join(output_dir, "combined", "results_*.csv"),
    ]
    for pattern in pattern_list:
        for csv_path in glob.glob(pattern):
            try:
                frame = pd.read_csv(csv_path)
            except Exception:
                continue
            needed = {"method", "dataset", "seed"}
            if not needed.issubset(frame.columns):
                continue
            for _, row in frame.iterrows():
                if row.get("status") == "failed" or pd.notna(
                    row.get("error", None)
                ):
                    continue
                keys.add(
                    (
                        str(row["method"]),
                        str(row["dataset"]),
                        str(row.get("training_layers", "")),
                        int(row["seed"]),
                    )
                )
    return keys


def proposed_run_dir(job):
    label = _experiment_label(vars(job))
    return Path(job.output_dir) / job.dataset / label / f"seed{job.seed}"


def proposed_is_finished(job):
    return (proposed_run_dir(job) / "models" / "model_info.json").exists()


def proposed_is_stale(job):
    """A crashed run wrote tasks/<method>/config.json but never reached the
    final model_info.json save. The handler's overwrite guard would raise on
    resubmission, so these are excluded and reported for manual cleanup."""
    run_dir = proposed_run_dir(job)
    config_marker = run_dir / "tasks" / job.method / "config.json"
    finished_marker = run_dir / "models" / "model_info.json"
    return config_marker.exists() and not finished_marker.exists()


def baseline_is_finished(job, finished_keys):
    key = (
        job.method,
        job.dataset,
        training_layers_label(job.training_layers),
        job.seed,
    )
    return key in finished_keys


def run_jobs_with_per_job_flush(jobs, output_dir, max_workers):
    """Same shape as the runner's run_jobs_in_parallel, but writes one chunk
    CSV per finished job instead of every 10. Baselines have no per-run disk
    output, so buffering rows risks losing finished work to a Slurm timeout;
    a chunk per job caps the loss at the jobs still in flight."""
    import concurrent.futures
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    chunk_dir = os.path.join(output_dir, "chunks")
    final_file = os.path.join(
        output_dir, "combined", f"results_{timestamp}.csv"
    )
    os.makedirs(chunk_dir, exist_ok=True)
    os.makedirs(os.path.dirname(final_file), exist_ok=True)
    print(f"Running {len(jobs)} jobs with {max_workers} workers.")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        futures = {executor.submit(run_single_job, job): job for job in jobs}
        for index, future in enumerate(
            concurrent.futures.as_completed(futures)
        ):
            result = future.result()
            if result is None:
                continue
            chunk_path = os.path.join(
                chunk_dir, f"chunk_{index}_{timestamp}.csv"
            )
            pd.DataFrame([result]).to_csv(chunk_path, index=False)
            print(f"Progress: {index + 1}/{len(jobs)} -> {chunk_path}")

    chunk_files = glob.glob(
        os.path.join(chunk_dir, f"chunk_*_{timestamp}.csv")
    )
    if chunk_files:
        pd.concat(
            [pd.read_csv(f) for f in chunk_files], ignore_index=True
        ).to_csv(final_file, index=False)
        print(f"Combined {len(chunk_files)} chunks into {final_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", choices=sorted(PLANS), required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only", choices=["proposed", "baselines"], default=None
    )
    arguments = parser.parse_args()

    plan = PLANS[arguments.plan]
    proposed_jobs = build_proposed_jobs(plan, arguments.output_dir)
    baseline_jobs = build_baseline_jobs(plan, arguments.output_dir)
    if arguments.only == "proposed":
        baseline_jobs = []
    elif arguments.only == "baselines":
        proposed_jobs = []

    finished_keys = load_finished_baseline_keys(arguments.output_dir)
    pending, stale = [], []
    for job in proposed_jobs:
        if proposed_is_finished(job):
            continue
        (stale if proposed_is_stale(job) else pending).append(job)
    pending += [
        job
        for job in baseline_jobs
        if not baseline_is_finished(job, finished_keys)
    ]
    total = len(proposed_jobs) + len(baseline_jobs)

    if stale:
        print(f"{len(stale)} crashed run dirs found, NOT resubmitted (the")
        print("handler's overwrite guard would raise). Delete to rerun:")
        for job in stale:
            print(f"  rm -rf {proposed_run_dir(job)}")

    print(
        f"Plan '{arguments.plan}': {total} jobs total "
        f"({len(proposed_jobs)} proposed, {len(baseline_jobs)} baselines), "
        f"{total - len(pending) - len(stale)} already done, "
        f"{len(stale)} stale, {len(pending)} to run"
    )
    for job in pending:
        variant = getattr(job, "ablation_variant", "primary")
        block_rows = getattr(job, "block_rows", "-")
        print(
            f"  {job.method:<12} {job.dataset:<20} "
            f"tl={job.training_layers_name:<24} variant={variant:<25} "
            f"block_rows={block_rows} seed={job.seed}"
        )

    if arguments.dry_run or not pending:
        return

    torch.set_float32_matmul_precision("high")
    for dataset_name in sorted({job.dataset for job in pending}):
        job_match = next(job for job in pending if job.dataset == dataset_name)
        base_name = getattr(job_match, "base_dataset", dataset_name)
        try:
            dataset.load_data(base_name)
        except Exception as download_error:
            print(
                f"Warning during pre-download of {dataset_name}: {download_error}"
            )

    start_time = time.time()
    run_jobs_with_per_job_flush(
        pending, arguments.output_dir, arguments.max_workers
    )
    print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
