# GIFT-Eval ensemble replication

This directory contains the code used to reproduce the probabilistic ensemble
results on GIFT-Eval. For a minimal introduction to the public ensemble API, see
[`probabilistic_ensemble_getting_started.ipynb`](../probabilistic_ensemble_getting_started.ipynb).

## Environment

Use Python 3.12 and run these commands from the Granite TSFM repository root:

```bash
uv sync --python 3.12 --locked --extra dev
uv pip install --python .venv/bin/python git+https://github.com/SalesforceAIResearch/gift-eval.git
uv pip check --python .venv/bin/python
export GIFT_EVAL=/path/to/downloaded/gift-eval
```

The root project provides Granite TSFM and the notebook and test dependencies.
Install GIFT-Eval from its repository into that environment; its dependencies
are installed with it.
Run the commands below with `.venv/bin/python` so uv does not resync the root
environment and remove those additional packages.

## Configurations

[`experiment_config.json`](experiment_config.json) defines the ensemble recipes,
quantile levels, datasets, forecast terms, and runner defaults. The notebook and
CLI both load this file through `run_gift_eval.py`. CLI arguments or explicit
arguments to `run_evaluation()` override its defaults.

Restart a running notebook kernel after editing the configuration because it is
loaded when the runner is imported.

| Configuration | Members |
|---|---|
| `probability-ensemble-uniform-ibm-tsfm-pt` | Research and Granite PatchTST, research and Granite FlowState, TTM, and Granite PatchTST-r2 |
| `probability-ensemble-uniform-ibm-tsfm-granite-pt` | Granite PatchTST, FlowState, TTM, and PatchTST-r2 |

Both configurations use uniform probability-space aggregation (linear pooling).

## Running

Run a small smoke test before the full evaluation:

```bash
.venv/bin/python notebooks/hfdemo/gift_eval_ensemble/run_gift_eval.py \
  --model_name_config probability-ensemble-uniform-ibm-tsfm-pt \
  --datasets us_births/M \
  --out_dir notebooks/hfdemo/gift_eval_ensemble/replication_runs/smoke
```

Omit `--datasets` to run all configured datasets:

```bash
.venv/bin/python notebooks/hfdemo/gift_eval_ensemble/run_gift_eval.py \
  --model_name_config probability-ensemble-uniform-ibm-tsfm-pt \
  --out_dir notebooks/hfdemo/gift_eval_ensemble/replication_runs/full
```

The runner automatically selects CUDA when available, followed by MPS and CPU.
Use `--device cuda` or `--device cpu` to select a device explicitly.

Results are written to `<out_dir>/<configuration>/all_results.csv`. Task status
and errors are written to `execution_log.csv`. The runner continues after an
individual task fails, so inspect the execution log and confirm result coverage.

Use `--skip_processed` to resume an unchanged run and `--save-member-results` to
save metrics for each ensemble member without repeating inference. Use a fresh
output directory after changing the configuration, preprocessing, or environment.

## Notebook and tests

[`granite_ensemble_gift_eval.ipynb`](granite_ensemble_gift_eval.ipynb) provides the
same smoke-test and full-evaluation workflow through `run_evaluation()`.
Select the root `.venv` as its kernel. If it is not listed, register it with:

```bash
.venv/bin/python -m ipykernel install --user --name granite-gift-eval --display-name "Granite GIFT-Eval"
```

Run the benchmark tests with:

```bash
PYTHONPATH=notebooks/hfdemo/gift_eval_ensemble .venv/bin/python -m pytest -q notebooks/hfdemo/gift_eval_ensemble/tests
```

Some tests and evaluation runs download model checkpoints.
