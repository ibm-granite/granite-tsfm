# GIFT-Eval ensemble replication

Use Python 3.12 and run the commands below from this directory.

This directory contains the benchmark-only implementation used to reproduce
GIFT-Eval results. For the minimal public ensemble example, see the parent
directory's [`probabilistic_ensemble_getting_started.ipynb`](../probabilistic_ensemble_getting_started.ipynb).

## Environment

```bash
uv sync
```

The local `pyproject.toml` isolates benchmark dependencies. It does not install
Granite TSFM itself. From this directory, add the repository root to the Python
path and set the dataset location:

```bash
export PYTHONPATH="$(cd ../../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export GIFT_EVAL=/path/to/downloaded/gift-eval
```

Both PatchTST-r1 and r2 load directly from this checkout's `tsfm_public` package.
No separate PatchTST source directory or model-path environment variable is
required. Restart the notebook kernel after switching source checkouts because
Python caches imported model classes.

The benchmark environment must also provide the runtime dependencies required
by Granite TSFM. The benchmark pins pandas 2.1.4; verify compatibility before
combining it with the main package environment.

For JupyterLab, install the notebook tools into this environment:

```bash
uv pip install jupyterlab ipykernel ipywidgets
.venv/bin/python -m jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

Select this environment as the notebook kernel. The notebook imports `run_evaluation()` from the local runner. The CLI calls
the same function.

## Configurations

| Configuration | Members |
|---|---|
| `probability-ensemble-uniform-ibm-tsfm-pt` | Research and Granite PatchTST-r1, research and Granite FlowState-r1.1, TTM-r3-pt, Granite PatchTST-r2 |
| `probability-ensemble-uniform-ibm-tsfm-granite-pt` | Granite PatchTST-r1, FlowState-r1.1, TTM-r3, PatchTST-r2 |

Ensembles use uniform probability-space aggregation (linear pooling).
The benchmark uses GIFT-Eval's generated windows and the reference branch's
TTM scaling and imputation. There is no selectable TTM scaling mode.
The TTM revision is selected dynamically by `get_model` from the requested
context length and prediction length; it is intentionally not pinned.

## Running

Smoke test:

```bash
uv run python run_gift_eval.py \
  --model_name_config probability-ensemble-uniform-ibm-tsfm-pt \
  --datasets us_births/M \
  --patchtst-use-fill-nan \
  --out_dir replication_runs/smoke \
  --skip_processed
```

Omit `--datasets` for the full benchmark. CUDA is selected automatically when
available, followed by MPS, then CPU. Set `--device cuda` or `--device cpu`
to select it explicitly. Match the device and NaN-filling flag used
for the reference results. `--patchtst-use-fill-nan` defaults to disabled.

Results are written to `<out_dir>/<configuration>/all_results.csv`, with task
status and errors in `execution_log.csv`. The runner continues after individual
task failures, so check coverage and the execution log before submission.
`--skip_processed` skips task names already present; use a fresh output directory
when changing configuration, preprocessing, or environment.

`--save-member-results` writes member metrics without repeating inference.
Additional options are `--seed` (42), `--out_name` (`all_results.csv`), and
`--error_log_name` (`execution_log.csv`).

## Notebook and tests

[`granite_ensemble_gift_eval.ipynb`](granite_ensemble_gift_eval.ipynb) calls `run_evaluation()`
for a smoke test and optionally the full benchmark. Use a batch job for runs
that exceed the interactive session limit. Cluster `scripts/` and local
`replication_runs/` should be excluded manually from commits.

All benchmark tests are in `test_ptm_forecasters.py`. The full suite includes
checkpoint downloads; the focused window tests use synthetic inputs:

```bash
uv run pytest test_ptm_forecasters.py
uv run pytest test_ptm_forecasters.py -k RunGiftEvalTest
```

For public release, pin dependency, Granite, GIFT-Eval, dataset, and checkpoint
revisions, and verify the notebook in a clean public environment. Contribute the
results CSV and `config.json` following the GIFT-Eval submission instructions,
with a link to the public replication code.
