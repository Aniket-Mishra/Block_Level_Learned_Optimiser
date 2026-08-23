# Block-Level Learned Optimization for Continual Learning

Code for the proposed method, experiment runner, and continual-learning baselines used in the thesis.

## Layout

- `experiments/run_plans.py`: main experiment entry point and editable plans.
- `experiments/exp_runner_1_1_config.py`: datasets, models, budgets, and method settings.
- `experiments/methods/`: proposed method.
- `experiments/models/`: base models, task encoders, and transformer.
- `experiments/baselines/`: baseline implementations.
- `experiments/optimal_baselines/`: baseline hyperparameter sweeps.
- `pretrained_task_encoders/`: required MNIST task-encoder checkpoint.
- `scripts/run_snellius.sh`: generic Snellius launcher.

Runtime code remains in `experiments/` to preserve the thesis implementation. It can be moved into `src/block_level_learned_optimization/` after the thesis when there is time to update imports and paths carefully.

## Environment

Python 3.13 and dependencies are in `uv.lock`.

Using `uv`:

```bash
uv sync
source .venv/bin/activate
```

Without `uv`:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-snellius.txt
```

## Configure experiments

Edit `PLANS` in `experiments/run_plans.py`. Dataset and model defaults are in `experiments/exp_runner_1_1_config.py`.

Available plan names:

```bash
cd experiments
python run_plans.py --help
```

## Run locally

Proposed method example:

```bash
cd experiments
python -u run_plans.py --plan mnist --only proposed --max-workers 1 --output-dir ../outputs
```

Baseline example:

```bash
cd experiments
python -u run_plans.py --plan cifar --only baselines --max-workers 1 --output-dir ../outputs
```

Omit `--only` to run every method defined by the selected plan.

Baseline sweep:

```bash
cd experiments
python -u -m optimal_baselines.run_sweep --max-workers 1 --output-dir ../outputs/optimal_baselines
```

Resume an interrupted proposed-method run:

```bash
cd experiments
python -u resume_run.py --run-dir PATH_TO_RUN
```

MNIST and CIFAR download automatically into `data/`. TinyImageNet also downloads automatically. Torchvision task encoders download pretrained weights on first use, so the machine needs network access or a populated PyTorch cache.

## Run on Snellius

Create `.venv` once on the login node using the environment commands above. Submit from the repository root:

```bash
sbatch --export=ALL,PLAN=mnist,ONLY=proposed,MAX_WORKERS=1,OUTPUT_DIR=/projects/PROJECT_NAME/outputs scripts/run_snellius.sh
```

Set `ONLY` to `proposed`, `baselines`, or `all`. Change Slurm time, partition, and worker count for the selected experiment.
