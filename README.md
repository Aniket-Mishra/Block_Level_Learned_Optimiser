# Block-Level Learned Optimisation for Continual Learning

Code for the proposed method, experiment runner, continual-learning baselines, and updated library with tests used in the thesis.

##### This is the final code for my Master's Thesis
##### Grade: 9/10

## Layout

- `experiments/run_plans.py`: main experiment entry point and editable plans.
- `experiments/exp_runner_1_1_config.py`: datasets, models, budgets, and method settings.
- `experiments/methods/`: compatibility imports for the proposed method.
- `experiments/models/`: base models, pretrained task encoders, and compatibility imports.
- `experiments/baselines/`: baseline implementations.
- `experiments/optimal_baselines/`: baseline hyperparameter sweeps.
- `src/block_level_learned_optimization/parameter_scope.py`: shared parameter selection and block sizing.
- `src/block_level_learned_optimization/optimizer_setup.py`: base-model and meta-optimizer construction.
- `src/block_level_learned_optimization/transformer.py`: transformer that predicts block updates.
- `src/block_level_learned_optimization/task_encoder.py`: generic support-input encoder.
- `src/block_level_learned_optimization/training_utils.py`: numerical helpers, learning-rate schedule and stopping controller.
- `src/block_level_learned_optimization/trainer.py`: meta-training and support/query adaptation.
- `examples/train_custom_data.py`: train and reload using a caller's model and data.
- `pretrained_task_encoders/`: required MNIST task-encoder checkpoint.
- `scripts/run_snellius.sh`: generic Snellius launcher.

Core optimizer code lives in `src/block_level_learned_optimization/`. Built-in datasets, model presets, experiment runners, aggregate metrics and baselines remain in `experiments/`. Existing experiment imports continue to work.

The trainer and learned update model are imported directly:

```python
from block_level_learned_optimization.trainer import PROPOSED
from block_level_learned_optimization.transformer import TransformerModel
```

`PROPOSED` takes a base model, transformer, loss, configuration and `batch_generator_class`. The sampler takes `(data, config_params)` and provides `get_batch(device=None)`, returning `x_sp`, `y_sp`, `x_qr` and `y_qr` for support and query examples. The experiment runner gives its existing `dataset.BatchGenerator`.

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
python -m pip install --no-deps -e .
```

## Train with your own data

Run the self-contained CPU example from the repository root:

```bash
python examples/train_custom_data.py --output-dir outputs/custom_data
```

We use a new output directory for each run. The example uses synthetic data cuz I am on laptop with no more access to Snellius, train two tasks and writes `checkpoint.pth`. It then reloads the saved state and checks if predictions and support-adaptation results match. The two test steps include a query measurement after one support adaptation.

Replace `Classifier` and the `(inputs, targets)` task pairs with new models and data. `EpisodeSampler` selects disjoint support and query examples. `model.pred_with_transformer` is sued for the parameter names that receive learned updates. The example selects `features.weight` and leaves its classifier on Adam. The generic task encoder fixes its input width on first use, so initialize it with your input shape before constructing the trainer.

The checkpoint example uses the same model and input shape, and Adam with EMA disabled. It includes the original model state because update scales depend on initialization statistics. It shows reconstruction after a completed task. The interrupted-run resume and transfer to different architectures need more work.

## Configure experiments

Edit `PLANS` in `experiments/run_plans.py`. Dataset and model defaults are in `experiments/exp_runner_1_1_config.py`.

Available plan names:

```bash
cd experiments
python run_plans.py --help
```

## Run locally

SplitMNIST test run from the root (seed 0, 5 tasks, 10 steps, 1warmup n 1 test):

```bash
python -u experiments/run_plans.py --plan splitmnist_smoke --only proposed --max-workers 1 --output-dir outputs/splitmnist_smoke
```

This checks execution rather than accuracy. With `test_steps=1`, the evaluator records query accuracy before its support adaptation. Use a fresh output directory to repeat a completed smoke run.

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
