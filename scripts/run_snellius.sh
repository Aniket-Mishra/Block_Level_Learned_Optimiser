#!/bin/bash
#SBATCH --job-name=block-learned-opt
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=120:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

repository_directory="${REPOSITORY_DIRECTORY:-$SLURM_SUBMIT_DIR}"
plan_name="${PLAN:-mnist}"
run_selection="${ONLY:-proposed}"
maximum_workers="${MAX_WORKERS:-1}"
output_directory="${OUTPUT_DIR:-$repository_directory/outputs}"

cd "$repository_directory"

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv. Create it on the login node before submitting."
    exit 1
fi

source .venv/bin/activate
cd experiments

runner_arguments=(
    --plan "$plan_name"
    --max-workers "$maximum_workers"
    --output-dir "$output_directory"
)

if [[ "$run_selection" == "proposed" || "$run_selection" == "baselines" ]]; then
    runner_arguments+=(--only "$run_selection")
elif [[ "$run_selection" != "all" ]]; then
    echo "ONLY must be proposed, baselines, or all."
    exit 1
fi

python -u run_plans.py "${runner_arguments[@]}"
