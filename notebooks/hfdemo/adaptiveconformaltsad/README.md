# Adaptive Conformal Anomaly Detection with Time Series Foundation Models

This repository contains the official implementation of the paper:

**"Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring"**  
*Accepted at ICLR 2026*

## Overview

This work introduces W1ACAS (Wasserstein-1 Adaptive Conformal Anomaly Scoring), a novel framework that combines time series foundation models with adaptive conformal prediction for robust anomaly detection in signal monitoring applications.

## Installation

### Required Dependencies

**Python Version:** 3.12 (recommended)

**Mandatory Installation:**
```bash
# Install granite-tsfm (Time Series Foundation Model)
pip install granite-tsfm
```

### Optional Dependencies

**For Chronos Model Support:**
```bash
pip install chronos-forecasting
```

**For TiRex Model Support:**
```bash
pip install tirex-ts
```

**For TSB-AD Evaluation Metrics:**
```bash
# Clone and install TSB-AD repository
git clone https://github.com/TheDatumOrg/TSB-AD.git
cd TSB-AD
pip install -e .
```

Note: TSB-AD is only required if you want to use the advanced evaluation metrics in `tsb_ad_evaluation.py`.

## Datasets

The paper evaluates the method on the following benchmark datasets:

- **YAHOO** (Laptev et al., 2015)
- **NEK** (Si et al., 2024)
- **NAB** (Ahmad et al., 2017)
- **MSL** (Lai et al., 2021)
- **IOPS** (IOPS, n.d.)
- **STOCK** (Tran et al., 2016)
- **WSD** (Zhang et al., 2022)
- **TAO** (Laboratory, 2024)
- **GECCO** (Rehbach et al., 2018)
- **LTDB** (Goldberger et al., 2000)
- **Genesis** (von Birgelen & Niggemann, 2018)

**Dataset Download:**  
All datasets should be downloaded from the [TSB-AD repository](https://github.com/TheDatumOrg/TSB-AD), which provides curated versions of these benchmarks. Follow the TSB-AD instructions to download and prepare the datasets.

An example dataset is provided in the `dataset/` folder for quick testing.

## Repository Structure

```
AdaptiveConformalTSAD/
├── README.md                          # This file
├── main_acas_w1.py                    # Main script for running W1ACAS
├── utils.py                           # Utility functions (plotting, context building)
├── standard_evaluation.py             # Standard evaluation metrics
├── tsb_ad_evaluation.py              # TSB-AD evaluation metrics (optional)
├── forecasters/                       # Forecasting model wrappers
│   ├── __init__.py
│   ├── granite_tsfm_forecaster.py    # Granite TSFM (TTM, FlowState)
│   ├── chronos_forecaster.py         # Chronos model wrapper
│   └── tirex_forecaster.py           # TiRex model wrapper
├── dataset/                           # Example datasets
│   └── 672_YAHOO_id_122_WebService_tr_500_1st_857.csv
└── output/                            # Output directory (created automatically)
    └── [results organized by dataset/model/config]
```

## Usage

### Basic Usage

Run W1ACAS with default parameters:

```bash
python main_acas_w1.py
```

Or specify a model:

```bash
# Using TTM model
python main_acas_w1.py --model_name ttm

# Using Chronos Bolt Small model
python main_acas_w1.py --model_name chronos-bolt-small

# Using TiRex model
python main_acas_w1.py --model_name tirex
```

### Custom Configuration

Run with specific model and parameters:

```bash
python main_acas_w1.py \
    --dataset_file_path dataset/672_YAHOO_id_122_WebService_tr_500_1st_857.csv \
    --model_name ttm \
    --context_length 512 \
    --prediction_length 10 \
    --aggregation_forecast_horizon HMC
```

### Available Models

- `ttm`: Granite Time Series Transformer (TTM-R2)
- `flowstate`: Granite FlowState
- `chronos-bolt-small`: Chronos Bolt Small (requires chronos installation)
- `tirex`: TiRex model (requires tirex installation)

### Command-Line Arguments

**Dataset & Model:**
- `--dataset_file_path`: Path to dataset CSV file
- `--model_name`: Model to use (ttm, flowstate, chronos-bolt-small, tirex)
- `--context_length`: Context window length (default: 90)
- `--prediction_length`: Forecast horizon (default: 15)

**W1ACAS Parameters:**
- `--significance_level`: Target significance level α (default: 0.01)
- `--aggregation_forecast_horizon`: P-value aggregation method (default: Cauchy)
  - Options: median, mean, min, max, Fisher, HMC, Tippett, Cauchy
- `--nonconformity_score`: Error metric (default: absolute_error)
- `--n_epochs`: Optimization epochs (default: 1)
- `--n_batch_update`: Batch size for weight updates (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--prior_past_weights_value`: Weight initialization (default: proximity)
- `--return_weights`: Return calibration weights (flag)

### View All Options

```bash
python main_acas_w1.py --help
```

## Output

Results are automatically saved in a hierarchical directory structure:

```
output/
└── {dataset_name}/
    └── {model_name}_{context_length}_{prediction_length}/
        └── W1ACAS_{nonconformity_score}_{aggregation_features}/
            ├── p_values.csv                          # P-values and predictions
            ├── evaluation.json                       # Evaluation metrics
            └── anomaly_detection_visualization.png   # Visualization plot
```

**Example output location:**
```
output/672_YAHOO_id_122_WebService_tr_500_1st_857/ttm_90_15/W1ACAS_absolute_error_Cauchy/
```

## Example Results

An example output is provided in the `output/` directory demonstrating the expected results format and visualizations.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{w1acas2026,
  title={Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring},
  author={[Authors]},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## References

**Foundation Models:**
- Ekambaram, V., et al. (2024). "Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series." arXiv preprint arXiv:2401.03955.
- Ansari, A. F., et al. (2024). "Chronos: Learning the Language of Time Series." arXiv preprint arXiv:2403.07815.
- Woo, G., et al. (2024). "TiRex: Time Series Representation Learning with Extreme Compression." arXiv preprint.

**Benchmarks:**
- Paparrizos, J., et al. (2022). "TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection." Proceedings of the VLDB Endowment, 15(8), 1697-1711.
- Laptev, N., et al. (2015). "Time-series extreme event forecasting with neural networks at Uber."
- Ahmad, S., et al. (2017). "Unsupervised real-time anomaly detection for streaming data."
- Lai, K. H., et al. (2021). "Revisiting time series outlier detection: Definitions and benchmarks."
- Si, Y., et al. (2024). "NEK: A new benchmark for time series anomaly detection."
- Tran, L., et al. (2016). "Stock market anomaly detection."
- Zhang, Y., et al. (2022). "WSD: A new benchmark for time series anomaly detection."
- Goldberger, A. L., et al. (2000). "PhysioBank, PhysioToolkit, and PhysioNet."
- Rehbach, F., et al. (2018). "GECCO: Benchmark for time series anomaly detection."
- von Birgelen, A., & Niggemann, O. (2018). "Genesis: A benchmark for time series anomaly detection."
- Laboratory, P. M. A. (2024). "TAO: Tropical Atmosphere Ocean project."
