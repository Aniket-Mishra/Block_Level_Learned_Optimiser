import argparse
import glob
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch

import test
import utils
from exp_runner_1_1_config import DEFAULT_OUTPUT_DIR
from exp_runner_1_1_fn_setup import (
    _RESULT_ROW_KEYS,
    _experiment_label,
    _init_method,
    _load_data,
    _load_task_scores,
    _profile_summary,
    _set_reproducibility,
    _training_layers_label,
    _update_task_scores,
    base_dataset_name,
)
from profiler import Profiler


def find_incomplete_runs(output_directory):
    incomplete = []
    for config_path in glob.glob(
        f"{output_directory}/**/tasks/proposed/config.json", recursive=True
    ):
        seed_directory = Path(config_path).parents[2]
        if not (seed_directory / "models" / "model_info.json").exists():
            incomplete.append(seed_directory)
    return incomplete


def last_completed_task(task_directory):
    accuracies = json.loads(
        (task_directory / "test_accuracy.json").read_text()
    )
    return max(int(task) for task in accuracies)


def load_accuracies(path):
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {
        int(task): {
            int(evaluated): float(value) for evaluated, value in scores.items()
        }
        for task, scores in raw.items()
    }


def newest_checkpoint(task_directory, task):
    candidates = glob.glob(
        str(task_directory / f"model_checkpoint_task{task}_step*.pth")
    )
    return max(
        candidates,
        key=lambda path: int(re.search(r"_step(\d+)\.pth$", path).group(1)),
    )


def restore_method(
    config, task_directory, completed_task, number_of_tasks, device
):
    checkpoint = torch.load(
        newest_checkpoint(task_directory, completed_task), map_location=device
    )
    method = _init_method(
        config, number_of_tasks, device, base_name=base_dataset_name(config)
    )
    method.model.load_state_dict(checkpoint["model_state_dict"])
    method.transformer_model.load_state_dict(
        checkpoint["transformer_state_dict"]
    )
    method.transformer_optimizer.load_state_dict(
        checkpoint["transformer_optimizer_state_dict"]
    )
    if (
        method.model_optimizer is not None
        and "model_optimizer_state_dict" in checkpoint
    ):
        method.model_optimizer.load_state_dict(
            checkpoint["model_optimizer_state_dict"]
        )

    # previous_tsk_score is the only cross-task state absent from the checkpoint
    # it equals the last task's importance, which task_scores.pkl already holds.
    scores_by_task = _load_task_scores(task_directory / "task_scores.pkl")
    method.previous_tsk_score = {
        name: tensor.to(device)
        for name, tensor in scores_by_task[completed_task].items()
    }
    return method, scores_by_task


def output_directory_for(seed_directory):
    """output_dir/<dataset>/<exp_label>/seed<N> -> output_dir."""
    return seed_directory.parents[2]


def row_already_written(output_directory, dataset, exp_label, seed):
    patterns = [
        str(output_directory / "chunks" / "chunk_*.csv"),
        str(output_directory / "combined" / "results_*.csv"),
    ]
    for pattern in patterns:
        for csv_path in glob.glob(pattern):
            frame = pd.read_csv(csv_path)
            if not {"dataset", "exp_label", "seed"}.issubset(frame.columns):
                continue
            match = (
                (frame["dataset"] == dataset)
                & (frame["exp_label"] == exp_label)
                & (frame["seed"] == seed)
            )
            if match.any():
                return True
    return False


def transformer_param_split(transformer_state):
    """(task-encoder params, meta-optimizer params) from a state dict."""
    te_params = sum(
        tensor.numel()
        for name, tensor in transformer_state.items()
        if name.startswith("te_model.")
    )
    total = sum(tensor.numel() for tensor in transformer_state.values())
    return te_params, total - te_params


def checkpoint_index(task_directory):
    """task -> [(step, mtime)] sorted by step, from the saved checkpoints."""
    index = {}
    pattern = re.compile(r"model_checkpoint_task(\d+)_step(\d+)\.pth$")
    for path in task_directory.glob("model_checkpoint_task*_step*.pth"):
        match = pattern.search(path.name)
        if match is None:
            continue
        task, step = int(match.group(1)), int(match.group(2))
        index.setdefault(task, []).append((step, path.stat().st_mtime))
    for entries in index.values():
        entries.sort()
    return index


def reconstructed_task_times(task_directory):
    """Per-task wall seconds estimated from checkpoint file mtimes.

    The span between the first and last checkpoint covers (last - first)
    steps; scaling it to the task's full step count estimates the whole task.
    Estimate only, but checkpointing is periodic so the error is small.
    """
    times = {}
    for task, entries in checkpoint_index(task_directory).items():
        if len(entries) < 2:
            continue
        steps = [step for step, _ in entries]
        mtimes = [mtime for _, mtime in entries]
        covered_steps = steps[-1] - steps[0]
        if covered_steps <= 0:
            continue
        span_seconds = mtimes[-1] - mtimes[0]
        times[task] = span_seconds * steps[-1] / covered_steps
    return times


def load_task_profiles(task_directory):
    path = task_directory / "task_profiles.json"
    if not path.exists():
        return {}
    return {int(task): v for task, v in json.loads(path.read_text()).items()}


def save_task_profiles(task_profiles, task_directory):
    utils.save_object(
        {str(task): profile for task, profile in task_profiles.items()},
        task_directory / "task_profiles.json",
    )


def time_only_stats(times_by_task):
    values = list(times_by_task.values())
    return {
        "fit_time_s_mean": round(sum(values) / len(values), 2),
        "fit_time_s_max": round(max(values), 2),
        "fit_time_s_min": round(min(values), 2),
    }


def profile_columns(task_profiles, task_directory, seed):
    """GPU/CPU/time columns for the results row.

    Profiled tasks contribute real Profiler numbers (GPU peak, CPU delta,
    time). Tasks that ran in the crashed process contribute only wall time
    reconstructed from checkpoint mtimes; their GPU/CPU peaks are gone.
    fit_time_s_total_est always covers ALL tasks (profiled + reconstructed).
    """
    columns = {}
    reconstructed = {
        task: seconds
        for task, seconds in reconstructed_task_times(task_directory).items()
        if task not in task_profiles
    }

    if task_profiles:
        columns.update(_profile_summary(task_profiles, seed=seed))
        columns["profiled_tasks"] = ",".join(
            str(task) for task in sorted(task_profiles)
        )
        columns["profile_source"] = "profiler_partial"
    elif reconstructed:
        columns.update(time_only_stats(reconstructed))
        columns["profile_source"] = "checkpoint_mtimes_only"

    all_times = dict(reconstructed)
    all_times.update(
        {task: profile["time_s"] for task, profile in task_profiles.items()}
    )
    if all_times:
        columns["fit_time_s_total_est"] = round(sum(all_times.values()), 2)
    return columns


def build_results_row(
    config,
    metrics,
    model_info,
    base_params,
    te_params,
    meta_params,
    seed_directory,
    resumed_from,
):
    """Same shape as the row the handler returns to run_plans.py."""
    train_scope = model_info.get("base_model_train_scope") or {}
    block_info = model_info.get("base_model_block_info") or {}

    row = {key: config.get(key) for key in _RESULT_ROW_KEYS}
    row["training_layers"] = _training_layers_label(
        config.get("training_layers")
    )
    row["exp_label"] = _experiment_label(config)
    row["run_dir"] = str(seed_directory)
    row["n_plasticity_levels"] = (
        len(config["plasticity_scales"])
        if config.get("plasticity_scales")
        else 0
    )
    row["total_blocks"] = block_info.get("total_blocks")
    row["max_blocks_in_any_tensor"] = block_info.get(
        "max_blocks_in_any_tensor"
    )
    row["selected_nweights"] = train_scope.get("selected_total_parameters")
    row["selected_nlayers"] = train_scope.get("selected_parameter_tensors")
    row["training_scope_label"] = train_scope.get("training_scope_label")
    row["base_params"] = base_params
    row["task_encoder_params"] = te_params
    row["meta_optimizer_params"] = meta_params
    row["transformer_params"] = te_params + meta_params
    row["resumed_from_task"] = resumed_from
    row.update(metrics)
    return row


def write_results_row(row, output_directory):
    """Mirrors run_plans.run_jobs_with_per_job_flush: one chunk CSV plus a
    combined CSV, so downstream aggregation picks the resumed run up."""
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    chunk_dir = output_directory / "chunks"
    combined_dir = output_directory / "combined"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame([row])
    frame.to_csv(chunk_dir / f"chunk_resumed_{timestamp}.csv", index=False)
    combined_path = combined_dir / f"results_resumed_{timestamp}.csv"
    frame.to_csv(combined_path, index=False)
    print(f"Results row written: {combined_path}")


def backfill_row(seed_directory):
    """Writes the results row for a run that finished through an older resume
    (which saved metrics_resumed.json but no CSV row). Saved files only, no
    recomputation. Wall time comes from checkpoint mtimes; GPU/CPU peaks are
    only present if task_profiles.json exists."""
    task_directory = seed_directory / "tasks" / "proposed"
    metrics_path = task_directory / "metrics_resumed.json"
    if not metrics_path.exists():
        print(
            f"{seed_directory}: complete without metrics_resumed.json; the "
            "run finished normally, so run_plans.py already wrote its row."
        )
        return

    config = json.loads((task_directory / "config.json").read_text())
    output_directory = output_directory_for(seed_directory)
    exp_label = _experiment_label(config)
    seed = int(config["seed"])
    if row_already_written(output_directory, config["dataset"], exp_label, seed):
        print(f"{seed_directory}: results row already present.")
        return

    metrics = json.loads(metrics_path.read_text())
    model_directory = seed_directory / "models"
    model_info = json.loads(
        (model_directory / "model_info.json").read_text()
    )
    model_state = torch.load(
        model_directory / "model.pth", map_location="cpu"
    )
    transformer_state = torch.load(
        model_directory / "transformer_model.pth", map_location="cpu"
    )
    base_params = sum(tensor.numel() for tensor in model_state.values())
    te_params, meta_params = transformer_param_split(transformer_state)

    row = build_results_row(
        config,
        metrics,
        model_info,
        base_params,
        te_params,
        meta_params,
        seed_directory,
        model_info.get("resumed_from_task"),
    )
    row.update(
        profile_columns(load_task_profiles(task_directory), task_directory, seed)
    )
    write_results_row(row, output_directory)


def save_completion(
    method,
    config,
    seed_directory,
    task_directory,
    resumed_from,
    previous_tasks_accuracy,
    future_tasks_accuracy,
    number_of_tasks,
    seed,
    task_profiles,
):
    model_directory = seed_directory / "models"
    model_directory.mkdir(parents=True, exist_ok=True)
    torch.save(method.model.state_dict(), model_directory / "model.pth")
    torch.save(
        method.transformer_model.state_dict(),
        model_directory / "transformer_model.pth",
    )

    base_params = sum(
        parameter.numel() for parameter in method.model.parameters()
    )
    transformer_params = sum(
        parameter.numel()
        for parameter in method.transformer_model.parameters()
    )
    metrics = utils.compute_all_metrics(
        previous_tasks_acc=previous_tasks_accuracy,
        future_tasks_acc=future_tasks_accuracy,
        n_tasks=number_of_tasks,
        base_params=base_params,
        transformer_params=transformer_params,
        bootstrap_seed=seed,
    )
    utils.save_object(metrics, task_directory / "metrics_resumed.json")

    model_info = {
        "base_model_train_scope": getattr(
            method, "base_model_train_scope", {}
        ),
        "base_model_block_info": getattr(method, "base_model_block_info", {}),
        "base_model_pred_with_transformer_tokens": getattr(
            method, "base_model_pred_with_transformer_tokens", []
        ),
        "resumed_from_task": resumed_from,
    }
    # model_info.json is the completion marker, so write it after everything else.
    utils.save_object(model_info, model_directory / "model_info.json")

    # run_plans.py writes the chunk/combined row from the handler's return
    # value, which a resumed run never produces; write the same row here.
    te_params = sum(
        parameter.numel()
        for parameter in method.transformer_model.te_model.parameters()
    )
    row = build_results_row(
        config,
        metrics,
        model_info,
        base_params,
        te_params,
        transformer_params - te_params,
        seed_directory,
        resumed_from,
    )
    row.update(profile_columns(task_profiles, task_directory, seed))
    write_results_row(row, output_directory_for(seed_directory))

    print(f"average_accuracy_final={metrics['average_accuracy_final']}")
    print(f"Complete: {model_directory / 'model_info.json'}")


def resume(run_path):
    # Accept either the seed directory or its tasks/proposed subdirectory.
    run_path = Path(run_path)
    if run_path.name == "proposed" and run_path.parent.name == "tasks":
        seed_directory = run_path.parents[1]
    else:
        seed_directory = run_path

    task_directory = seed_directory / "tasks" / "proposed"
    model_directory = seed_directory / "models"
    if (model_directory / "model_info.json").exists():
        print(f"{seed_directory}: already complete; checking the results row.")
        backfill_row(seed_directory)
        return

    config = json.loads((task_directory / "config.json").read_text())
    if config.get("use_ema"):
        raise SystemExit(
            f"{seed_directory}: use_ema runs are not resumable; the consolidated importance is not checkpointed."
        )

    seed = int(config["seed"])
    base_name = base_dataset_name(config)

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    _set_reproducibility(seed)
    device, available_gpus = utils.set_device()
    config["device"] = device

    train_datasets, test_datasets, task_labels, number_of_tasks = _load_data(
        config, base_name=base_name
    )
    completed_task = last_completed_task(task_directory)
    remaining_tasks = range(completed_task + 1, number_of_tasks)
    print(
        f"{config['dataset']} seed {seed}: {completed_task + 1} tasks done, running {list(remaining_tasks)}"
    )

    method, scores_by_task = restore_method(
        config, task_directory, completed_task, number_of_tasks, device
    )
    previous_tasks_accuracy = load_accuracies(
        task_directory / "test_accuracy.json"
    )
    future_tasks_accuracy = load_accuracies(
        task_directory / "future_accuracy.json"
    )
    # Accumulates across resumes: a task profiled by an earlier resume keeps
    # its real numbers even if this process crashes too.
    task_profiles = load_task_profiles(task_directory)

    evaluation_device = (
        device
        if base_name == "cifar100"
        else (
            f"cuda:{available_gpus[1]}" if len(available_gpus) > 1 else device
        )
    )

    for task_index in remaining_tasks:
        print(f"Task {task_index}")
        with Profiler(label=f"task_{task_index}") as profile:
            method, weights_info = method.fit(
                train_datasets[task_index], task_index, save_dir=task_directory
            )
        task_profiles[task_index] = profile.result.summary()
        save_task_profiles(task_profiles, task_directory)
        _update_task_scores(
            task_directory / "task_scores.pkl",
            task_index,
            weights_info["score"],
            scores_by_task,
        )
        previous_tasks_accuracy[task_index] = test.test_previous_tasks(
            method, test_datasets, task_index, evaluation_device
        )
        utils.save_object(
            previous_tasks_accuracy, task_directory / "test_accuracy.json"
        )
        if task_index + 1 < number_of_tasks:
            future_tasks_accuracy[task_index] = test.test_future_tasks(
                method,
                test_datasets,
                task_index,
                number_of_tasks,
                evaluation_device,
            )
            utils.save_object(
                future_tasks_accuracy, task_directory / "future_accuracy.json"
            )

    save_completion(
        method,
        config,
        seed_directory,
        task_directory,
        completed_task + 1,
        previous_tasks_accuracy,
        future_tasks_accuracy,
        number_of_tasks,
        seed,
        task_profiles,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Resume a crashed proposed-method run."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="A specific seed directory; skips auto-discovery.",
    )
    arguments = parser.parse_args()

    if arguments.run_dir:
        resume(Path(arguments.run_dir))
        return

    incomplete = find_incomplete_runs(arguments.output_dir)
    if not incomplete:
        print("No incomplete runs found.")
        return
    if len(incomplete) > 1:
        print("Several incomplete runs found; choose one with --run-dir:")
        for seed_directory in incomplete:
            print(f"  {seed_directory}")
        return
    resume(incomplete[0])


if __name__ == "__main__":
    main()
